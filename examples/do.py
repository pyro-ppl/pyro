counter = 0

for m in [1, 2, 3, 4, 5]:
    for lr in [0.01, 0.05]:
        counter +=1
        gpu = counter % 2
        print("CUDA_VISIBLE_DEVICES=%d python hmm.py --cuda -m %d -lr %.3f" % (gpu, m, lr))
