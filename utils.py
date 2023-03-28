from subprocess import Popen, PIPE, STDOUT


def run_process(args, shell=False, **kwargs):
    kwargs.update({
        'args': args,
        'shell': shell,
        'stdout': PIPE,
        'stderr': STDOUT,
        'close_fds': True
    })
    process = Popen(**kwargs)
    for line in iter(process.stdout.readline, b''):
        print(line.rstrip().decode('utf-8'))
    process.stdout.close()
    return process.wait()
