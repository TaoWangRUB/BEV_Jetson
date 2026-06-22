/* mpu9250_reader — configure + stream the J106 onboard MPU-9250 (SPI).
 *
 * The IMU is on /dev/spidev1.0 (Tegra186 SPI3, CS0). This wakes it, sets the
 * DLPF / sample rate / full-scale ranges, then burst-reads accel+gyro, scales to
 * SI units, timestamps each sample (CLOCK_MONOTONIC), and prints CSV:
 *     t_ns,ax,ay,az,gx,gy,gz      (accel m/s^2, gyro rad/s)
 * This is the basis for the cuVSLAM IMU feed (RegisterImuMeasurement).
 *
 *   gcc mpu9250_reader.c -o mpu9250_reader
 *   sudo ./mpu9250_reader [device] [rate_hz]      # defaults /dev/spidev1.0 200
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

enum {
    SMPLRT_DIV = 0x19, CONFIG = 0x1A, GYRO_CONFIG = 0x1B, ACCEL_CONFIG = 0x1C,
    ACCEL_CONFIG2 = 0x1D, PWR_MGMT_1 = 0x6B, WHO_AM_I = 0x75, ACCEL_XOUT_H = 0x3B
};

static int fd;
static uint32_t speed = 1000000;  /* 1 MHz: MPU-9250 SPI register-read max */

static void xfer(uint8_t *tx, uint8_t *rx, int n) {
    struct spi_ioc_transfer tr;
    memset(&tr, 0, sizeof(tr));
    tr.tx_buf = (unsigned long)tx; tr.rx_buf = (unsigned long)rx;
    tr.len = n; tr.speed_hz = speed; tr.bits_per_word = 8;
    ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
}
static uint8_t rd(uint8_t reg) { uint8_t tx[2] = {reg | 0x80, 0}, rx[2] = {0, 0}; xfer(tx, rx, 2); return rx[1]; }
static void wr(uint8_t reg, uint8_t val) { uint8_t tx[2] = {reg & 0x7F, val}, rx[2]; xfer(tx, rx, 2); }
static int16_t be16(uint8_t hi, uint8_t lo) { return (int16_t)((hi << 8) | lo); }

int main(int argc, char **argv) {
    const char *dev = argc > 1 ? argv[1] : "/dev/spidev1.0";
    int rate = argc > 2 ? atoi(argv[2]) : 200;
    fd = open(dev, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    uint8_t mode = 0, bits = 8;
    ioctl(fd, SPI_IOC_WR_MODE, &mode);
    ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);

    fprintf(stderr, "WHO_AM_I=0x%02X (expect 0x71)\n", rd(WHO_AM_I));

    /* configure: PLL clock + awake; gyro/accel DLPF ~41 Hz; rate; FS ±500 dps / ±4 g */
    wr(PWR_MGMT_1, 0x01); usleep(10000);
    wr(CONFIG, 0x03);
    int div = 1000 / rate - 1; if (div < 0) div = 0;
    wr(SMPLRT_DIV, (uint8_t)div);   /* output rate = 1000/(1+div) Hz */
    wr(GYRO_CONFIG, 0x08);          /* ±500 dps */
    wr(ACCEL_CONFIG, 0x08);         /* ±4 g */
    wr(ACCEL_CONFIG2, 0x03);
    usleep(20000);

    const double A = (4.0 / 32768.0) * 9.80665;                 /* LSB -> m/s^2 */
    const double G = (500.0 / 32768.0) * (3.14159265358979 / 180.0); /* LSB -> rad/s */
    long period = 1000000000L / rate;
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    printf("t_ns,ax,ay,az,gx,gy,gz\n");
    for (;;) {
        uint8_t tx[15] = {ACCEL_XOUT_H | 0x80}, rx[15] = {0};
        xfer(tx, rx, 15);                 /* accel(6) temp(2) gyro(6) */
        struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
        long long t_ns = (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
        double ax = be16(rx[1], rx[2]) * A, ay = be16(rx[3], rx[4]) * A, az = be16(rx[5], rx[6]) * A;
        double gx = be16(rx[9], rx[10]) * G, gy = be16(rx[11], rx[12]) * G, gz = be16(rx[13], rx[14]) * G;
        printf("%lld,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f\n", t_ns, ax, ay, az, gx, gy, gz);
        fflush(stdout);
        next.tv_nsec += period;
        while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
}
