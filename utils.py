from subprocess import Popen, PIPE, STDOUT


def run_process(args, shell=False, **kwargs):
    kwargs.update({
        'args': args,
        'shell': shell,
        'stdout': PIPE,
        'stderr': STDOUT,
        'close_fds': True
    })
    process = Popen(bufsize=-1,**kwargs)
    char = b""
    for b in iter(lambda: process.stdout.read(1), b''):
        char += b
        try:
            print(char.decode('utf-8'), end='')
            char = b""
        except:
            pass
    process.stdout.close()
    return process.wait()
