from invoke import task, run

@task
def clean():
    run("rm -f out.rs")
    run("rm -f out")
    print("clean!")

@task
def compile():
    run("./target/debug/stack_arithmetic")
    print("first stage compiled `out.rs`!")
    run("rustc out.rs", pty=True)
    print("second stage compiled `out`!")
