date
rm 2-methyl-hexane_unrestrained_convergence.csv
cargo build --release
for i in $(seq 10000 10000 1000000); do
    target/release/nn_entropy $i >> 2-methyl-hexane_unrestrained_convergence.csv
done
date