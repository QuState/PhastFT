use phastft::{
    fft_64, fft_64_dit, fft_64_with_opts_and_plan,
    options::Options,
    planner::{Direction, Planner64},
};

/// This example demonstrates the bit reversal behavior of DIF and DIT FFT algorithms
/// and how to control bit reversal in DIF FFT.
fn main() {
    println!("PhastFT Bit Reversal Control Example");
    println!("=====================================\n");

    demonstrate_dif_bit_reversal();
    println!();
    demonstrate_dit_bit_reversal();
    println!();
    demonstrate_chaining_ffts();
}

fn demonstrate_dif_bit_reversal() {
    println!("DIF FFT Bit Reversal Behavior");
    println!("------------------------------");

    let size = 8;
    let mut reals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut imags = vec![0.0; size];

    // Save original for comparison
    let original_reals = reals.clone();

    // Standard DIF FFT with bit reversal (default)
    println!("Input (normal order): {:?}", original_reals);
    fft_64(&mut reals, &mut imags, Direction::Forward);
    println!(
        "Output (bit-reversed): First 4 real components: [{:.2}, {:.2}, {:.2}, {:.2}]",
        reals[0], reals[1], reals[2], reals[3]
    );

    // DIF FFT without bit reversal
    let mut reals_no_br = original_reals.clone();
    let mut imags_no_br = vec![0.0; size];

    let mut opts = Options::default();
    opts.dif_perform_bit_reversal = false; // Skip bit reversal
    let planner = Planner64::new(size, Direction::Forward);

    fft_64_with_opts_and_plan(&mut reals_no_br, &mut imags_no_br, &opts, &planner);
    println!("Output (decimated order, no bit reversal): First 4 real components: [{:.2}, {:.2}, {:.2}, {:.2}]",
             reals_no_br[0], reals_no_br[1], reals_no_br[2], reals_no_br[3]);

    println!("\nNote: With dif_perform_bit_reversal=false, output stays in decimated order.");
    println!("This is useful when chaining operations or when bit-reversed output is not needed.");
}

fn demonstrate_dit_bit_reversal() {
    println!("DIT FFT Bit Reversal Behavior");
    println!("------------------------------");

    let size = 8;
    let mut reals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut imags = vec![0.0; size];

    println!("Input (normal order): {:?}", reals);

    // DIT FFT always bit-reverses the input internally
    fft_64_dit(&mut reals, &mut imags, Direction::Forward);

    println!(
        "Output (normal order): First 4 real components: [{:.2}, {:.2}, {:.2}, {:.2}]",
        reals[0], reals[1], reals[2], reals[3]
    );

    println!("\nNote: DIT always bit-reverses the input internally.");
    println!("The Options::dif_perform_bit_reversal flag does not affect DIT.");
}

fn demonstrate_chaining_ffts() {
    println!("Chaining FFT Operations");
    println!("------------------------");

    let size = 8;
    let mut reals = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut imags = vec![0.0; size];

    // Create options to skip bit reversal in DIF
    let mut opts_no_br = Options::default();
    opts_no_br.dif_perform_bit_reversal = false;

    let planner_forward = Planner64::new(size, Direction::Forward);
    let planner_reverse = Planner64::new(size, Direction::Reverse);

    println!("Original signal: {:?}", reals);

    // Forward FFT without bit reversal
    fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts_no_br, &planner_forward);
    println!(
        "After forward FFT (no bit reversal): [{:.2}, {:.2}, {:.2}, {:.2}, ...]",
        reals[0], reals[1], reals[2], reals[3]
    );

    // Inverse FFT without bit reversal (processes decimated input correctly)
    fft_64_with_opts_and_plan(&mut reals, &mut imags, &opts_no_br, &planner_reverse);
    println!(
        "After inverse FFT (no bit reversal): [{:.2}, {:.2}, {:.2}, {:.2}, ...]",
        reals[0], reals[1], reals[2], reals[3]
    );

    println!("\nWhen chaining DIF FFTs without intermediate processing,");
    println!("skipping bit reversal can improve performance.");
}
