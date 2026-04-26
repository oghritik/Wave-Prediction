# test_tcp_init.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Note: We pass the master address and port to the run function now
def run(rank, size, master_addr, master_port):
    """ The target function that will be run on each process. """
    print(f"--> Starting process on Rank {rank}.")

    # Set the GLOO transport environment variable within the spawned process
    os.environ['GLOO_DEVICE_TRANSPORT'] = 'tcp'
    
    # Construct the init_method string
    init_method = f'tcp://{master_addr}:{master_port}'

    dist.init_process_group(
        backend='gloo',
        init_method=init_method,
        world_size=size,
        rank=rank
    )
    
    print(f"✅ Process group initialized successfully on Rank {rank}!")
    dist.destroy_process_group()
    print(f"--> Process on Rank {rank} finished cleanly.")

def main():
    print("--- Starting minimal DDP test with TCP init ---")
    world_size = 2
    master_addr = '127.0.0.1' # Use the explicit IP for localhost
    master_port = '29510'     # Use a different, less common port

    try:
        mp.spawn(run,
                 # Pass the address and port as arguments to the run function
                 args=(world_size, master_addr, master_port),
                 nprocs=world_size,
                 join=True)
        print("\nSUCCESS: Minimal DDP test completed.")
    except Exception as e:
        print(f"\nFAILED: The test script crashed. See error below.")
        # PyTorch often wraps the real error, so let's print the whole exception
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()