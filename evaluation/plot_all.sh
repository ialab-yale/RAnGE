alias safe_python="XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8"
plot_dir=$(pwd)/../figure_tests
mkdir $plot_dir || true &&
log=$plot_dir/plotting_output.txt &&

echo "Figure 1..." &&
safe_python exploration_and_value.py ../logs/demo_bimodal \
    $plot_dir/fig1 --sim_time 10 --fixed_t 0.01 >> $log &&

echo "Figure 2..." &&
safe_python two_distribution_exploration_comparison.py \
    ../logs/demo_uniform ../logs/demo_bimodal \
    $plot_dir/fig2 --x0 0.1 --v0 -0.1 >> $log && 

echo "Figure 3..." && 
safe_python trajectory_evolution.py ../logs/demo_bimodal \
    $plot_dir/fig3 --frames 5 --sim_time 20 --fixed_t 0.02 --noise wc >> $log &&

echo "Figure 4..." &&
safe_python batch_simulation/box_plots_1x4.py $plot_dir/fig4_1x4 >> $log &&
safe_python batch_simulation/box_plots_2x2.py $plot_dir/fig4_2x2 >> $log &&

echo "Figure 5..." &&
safe_python physical_experiments/plot_from_real_log.py physical_experiments/CONST_WIND.pickle \
    $plot_dir/fig5a "Constant Wind" >> $log &&
safe_python physical_experiments/plot_from_real_log.py physical_experiments/NO_WIND.pickle \
    $plot_dir/fig5b "No Disturbance" >> $log &&
safe_python physical_experiments/plot_from_real_log.py physical_experiments/VAR_WIND.pickle \
    $plot_dir/fig5c "Variable Wind" >> $log &&

echo "Movie..." &&
# For a full movie, set frames to 250 and sim_time to 25
safe_python movie.py ../logs/demo_bimodal $plot_dir/movie \
    --frames 10 --sim_time 5 --fixed_t 0.02 >> $log &&

echo "Multiplot..." &&
safe_python many_simulations.py ../logs/demo_bimodal $plot_dir/multi_simulate \
    --noise wc --x0_num 2 --v0_num 2 --fixed_t 0.02 --horizons 20 >> $log
