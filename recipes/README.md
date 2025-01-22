## How to run the tests:

1. Set the THORNLIST environment variable to point to a thornlist in Cactus's directory of thornlists, e.g.:
```bash
$ export THORNLIST=$HOME/Cactus/thornlists/einsteintoolkit.th
```

2. Call the script, e.g.:

```bash
$ bash ./recipes/ricci/build-and-run.sh 
```

The script will run the test interactively. This will not work on a machine
that requires you to use the queueing system.
