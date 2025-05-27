from numba import cuda

def select_gpu():
    """ Select the first available GPU (necessary on some multi-gpu systems). """
    for i in range(len(cuda.gpus)):
        try:
            cuda.select_device(i)
            print(f"Using device {i}")
            break
        except Exception:
            pass
    return
    
if __name__ == "__main__":
    select_gpu()
