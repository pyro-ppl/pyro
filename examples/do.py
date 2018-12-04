counter = 0

for m in [1, 2, 3, 4, 5]:
    for lr in [0.02, 0.08]:
        for cn in [10.0]:
            for b1 in [0.8, 0.9]:
                counter += 1
                gpu = counter % 2
                arg = "CUDA_VISIBLE_DEVICES=%d python hmm.py --cuda -m %d -lr %.3f -cn %.3f -b1 %.3f"
                print(arg % (gpu, m, lr, cn, b1))
