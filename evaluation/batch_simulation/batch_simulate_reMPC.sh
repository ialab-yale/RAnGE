(XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_bimodal reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise rau ;
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_bimodal reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise ran ;
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_bimodal reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise none ;
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_bimodal reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise wc ;
) &
(XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_uniform reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise rau; 
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_uniform reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise ran; 
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_uniform reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise none; 
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python3.8 gen_sims.py ../../logs/demo_uniform reMPC --x0_num 8 --v0_num 8 --fixed_t 0.02 --horizons 20 --noise wc; 
) &