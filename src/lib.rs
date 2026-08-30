//! # Wang-Mendel Fuzzy System (`wm_fuzzy`)
//!
//! A Rust implementation of the Wang-Mendel method for generating and evaluating
//! fuzzy rule bases from numerical and linguistic data.
//!
//! ## Quick Start
//!
//! ```rust
//! use std::collections::HashMap;
//! use wm_fuzzy::{Antecedents, FuzzyRule, TNorm, WMModel};
//!
//! let mut model = WMModel::new();
//! // Configure limits, generate rules, or load pre-existing rule bases.
//! ```

pub mod wm_model;

pub use wm_model::{
    ActivatedRule, Antecedents, EvaluationMetrics, FuzzyRule, Granularity, TNorm, WMModel,
    WMModelError,
};
