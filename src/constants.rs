use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(Debug, Clone, Copy, Display, EnumString, IntoStaticStr, AsRefStr)]
pub enum TypeOnnxModel {
    #[strum(serialize = "face")]
    Face,
    #[strum(serialize = "object")]
    Object,
}
