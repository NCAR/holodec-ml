import subprocess
import sys
import os 


def startProcess(name, path):
    """
    Starts a process in the background and writes a PID file

    returns integer: pid
    """

    # Check if the process is already running
    status, pid = processStatus(name)

    if status == RUNNING:
        raise AlreadyStartedError(pid)

    # Start process
    process = subprocess.run(path + ' > /dev/null 2> /dev/null &')

    # Write PID file
    pidfilename = os.path.join(PIDPATH, name + '.pid')
    pidfile = open(pidfilename, 'w')
    pidfile.write(str(process.pid))
    pidfile.close()

    return process.pid



model_config = sys.argv[1]

n_gpus = 1 
workers = 4
workers_per_gpu = workers // n_gpus

# if psutil.pid_exists(pid):

total = 0
for worker in range(workers_per_gpu):
    for work in range(n_gpus):
        
        if (total + 1) == workers:
            launchargs = f"python /glade/work/schreck/repos/HOLO/holodec-ml/holodecml/reader/resnet_inference_mp.py {model_config} {total} {work}"
        else:
            launchargs = f"python /glade/work/schreck/repos/HOLO/holodec-ml/holodecml/reader/resnet_inference_mp.py {model_config} {total} {work} &"
        
        print(launchargs)
        
        #os.system(launchargs)
        
        total += 1

