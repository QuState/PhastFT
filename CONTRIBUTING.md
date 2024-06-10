# CONTRIBUTING

Contributions are welcome, and they are greatly appreciated!

1. Fork the `phastft` repository
2. Clone your fork to your dev machine/environment:
```bash
git clone git@github.com:<username>/phastft.git
```
3. [Install Rust](https://www.rust-lang.org/tools/install) and setup [nightly](https://rust-lang.github.io/rustup/concepts/channels.html) Rust

4. Setup the git hooks by in your local repo:
```bash
cd PhastFT
git config core.hooksPath ./hooks 
```

5. When you're done with your changes, ensure the tests pass with:
```bash
cargo test --all-features
```

7. Commit your changes and push them to GitHub

8. Submit a pull request (PR) through the [GitHub website](https://github.com/QuState/phastft/pulls).

## Pull Request Guidelines

Before you submit a pull request, please check the following:
- The pull request should include tests if it adds and/or changes functionalities.
