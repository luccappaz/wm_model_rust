# `wm_fuzzy`

[![Crates.io](https://img.shields.io/crates/v/wm_fuzzy.svg)](https://crates.io/crates/wm_fuzzy)
[![Documentation](https://docs.rs/wm_fuzzy/badge.svg)](https://docs.rs/wm_fuzzy)
[![CI](https://github.com/your-username/wm_fuzzy/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/wm_fuzzy/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A high-performance Rust implementation of the **Wang-Mendel (WM) algorithm** for automated fuzzy rule generation, inference, and binary/continuous classification from numerical data.

---

## Features

* **Automated Rule Extraction:** Fast generation of fuzzy IF-THEN rules from numerical datasets via Cartesian partitioning.
* **Algebraic T-Norms:** Configurable firing strength aggregation using **Product** or **Minimum** ($t$-norms).
* **Robust Fallback Engine:** Out-of-distribution handling via continuous Euclidean distance mapping in the membership center space.
* **Native Serialization:** Full JSON support for saving and loading learned rule bases and evaluation metrics (`serde`).
* **Interactive CLI Prompt:** Terminal interface for inspecting fuzzy partitions, $[a, b, c]$ vertices, and querying rule activations.
* **Evaluation & Diagnostics:** Computes Confusion Matrix, Accuracy, Precision, Recall, and F1-Score directly.

---

## Installation

Add `wm_fuzzy` to your project's `Cargo.toml`:

```toml
[dependencies]
wm_fuzzy = "0.1.0"
```

---

## Quickstart

```rust
use std::collections::HashMap;
use wm_fuzzy::{Granularity, TNorm, WMModel, WMModelError};

fn main() -> Result<(), WMModelError> {
    let mut model = WMModel::new();

    // 1. Configure fuzzy partitions (Universes of Discourse)
    model.granularity.insert("temperature".to_string(), Granularity::Three);
    model.granularity.insert("pressure".to_string(), Granularity::Three);

    model.labels.insert(
        "temperature".to_string(),
        vec!["low".to_string(), "medium".to_string(), "high".to_string()],
    );
    model.labels.insert(
        "pressure".to_string(),
        vec!["low".to_string(), "medium".to_string(), "high".to_string()],
    );

    model.limits.insert("temperature".to_string(), vec![10.0, 25.0, 40.0]);
    model.limits.insert("pressure".to_string(), vec![80.0, 100.0, 120.0]);

    // 2. Prepare training data
    let mut x_train = Vec::new();
    let y_train = vec![0.1, 0.5, 0.9];

    let samples = [
        (12.0, 85.0),
        (24.0, 100.0),
        (38.0, 118.0),
    ];

    for &(t, p) in &samples {
        let mut row = HashMap::new();
        row.insert("temperature".to_string(), t);
        row.insert("pressure".to_string(), p);
        x_train.push(row);
    }

    let features = vec!["temperature".to_string(), "pressure".to_string()];

    // 3. Generate rules (min_support = 0.01, min_confidence = 0.1)
    let rules = model.generate_rules(&x_train, &y_train, &features, TNorm::Product, 0.01, 0.1);
    println!("Extracted {} fuzzy rules.", rules.len());

    // 4. Save learned rules to JSON
    model.save_rules("rules.json")?;

    // 5. Predict on test samples
    let (continuous_preds, binary_preds) = model.predict(
        &x_train,
        Some(0.5), // Classification threshold
        0.0,       // Fallback value
        TNorm::Product,
    )?;

    println!("Continuous predictions: {:?}", continuous_preds);
    println!("Binary predictions:     {:?}", binary_preds);

    // 6. Evaluate and write metrics to fuzzy_results.json
    let y_true = vec![0, 1, 1];
    let metrics = model.evaluate(&x_train, &y_true, None)?;
    println!("Model Accuracy: {:.2}%", metrics.accuracy * 100.0);

    Ok(())
}
```

---

## Mathematical Formulation

### 1. Triangular Membership Functions
For parameters $[a, b, c]$:
$$\mu(x) = \begin{cases} 
0 & x \le a \text{ or } x \ge c \\
\frac{x - a}{b - a} & a < x \le b \\
\frac{c - x}{c - b} & b < x < c 
\end{cases}$$

### 2. Rule Firing Strength ($t$-norm)
$$\alpha_k(x) = \prod_{i=1}^{n} \mu_{A_i^k}(x_i) \quad \text{or} \quad \min_{i=1 \dots n} \mu_{A_i^k}(x_i)$$

### 3. Defuzzification (Weighted Average)
$$\hat{y} = \frac{\sum_{k=1}^{R} \alpha_k \cdot \bar{y}_k}{\sum_{k=1}^{R} \alpha_k}$$

---

## CLI & Interactive Exploration

To launch the built-in interactive antecedent selector:

```rust
let user_antecedents = model.prompt_antecedents();
model.infer(&user_antecedents, TNorm::Product);
```

```text
==================================================
           FUZZY ANTECEDENTS SELECTION            
==================================================

Attribute: "temperature"
Select a linguistic term:
  [1] low          -> [a:   10.000, b (center):   10.000, c:   25.000]
  [2] medium       -> [a:   10.000, b (center):   25.000, c:   40.000]
  [3] high         -> [a:   25.000, b (center):   40.000, c:   40.000]
Enter option number (1-3) or exact name: 2
```

---

## Development & Testing

```bash
# Run unit and integration tests
cargo test --all-targets --all-features

# Run linter checks
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation locally
cargo doc --no-deps --open

# Run the demo example
cargo run --example demo
```

---

## License

* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
