from multiprocessing import Process, Value
from time import sleep

def do_work():
    print("Working...")
    sleep(1)

def worker(run):
    while True:
        if run.value:
            do_work()

if __name__ == "__main__":
    run = Value("i", 1)
    p = Process(target=worker, args=(run,))
    p.start()
    while True:
        input()
        run.value = not run.value