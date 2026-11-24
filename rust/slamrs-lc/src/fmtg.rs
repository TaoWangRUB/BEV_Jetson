//! C's `%.9g`, as numpy.savetxt writes the Python csvs: nine significant digits, trailing zeros
//! stripped, scientific notation below 1e-4 or from 1e9 with a two-digit exponent. Needed so the
//! VIO csv of this binary is byte-identical to run_slamrs_lc.py's.

pub fn g9(x: f64) -> String {
    gn(x, 9)
}

pub fn gn(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    // scientific with digits-1 decimals gives the rounded mantissa and the decimal exponent
    let sci = format!("{:.*e}", digits - 1, x);
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if exp < -4 || exp >= digits as i32 {
        let mant = strip_zeros(mant);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    } else {
        let decimals = (digits as i32 - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{:.*}", decimals, x))
    }
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::g9;

    #[test]
    fn matches_numpy() {
        assert_eq!(g9(0.0), "0");
        assert_eq!(g9(1.0), "1");
        assert_eq!(g9(796.0), "796");
        assert_eq!(g9(-0.352), "-0.352");
        assert_eq!(g9(0.000123), "0.000123");
        assert_eq!(g9(1e-5), "1e-05");
        assert_eq!(g9(3774614057000.0), "3.77461406e+12");
        assert_eq!(g9(123456789.0), "123456789");
        assert_eq!(g9(1234567890.0), "1.23456789e+09");
        assert_eq!(g9(0.1234567891234), "0.123456789");
        assert_eq!(g9(16.6412345678), "16.6412346");
    }
}
