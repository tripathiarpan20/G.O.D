## For dev without docker:

1. Run bootstrap.sh
```bash
sudo -E bash bootstrap.sh
source $HOME/.bashrc
source $HOME/.venv/bin/activate
```

2. Install dependencies
```bash
find . -path "./venv" -prune -o -path "./.venv" -prune -o -name "requirements.txt" -exec pip install -r {} \;
./install_axolotl.sh
pip install "git+https://github.com/rayonlabs/fiber.git@1.0.0#egg=fiber[full]"

```

3. Setup dev env

```bash
task dev_setup
```

4.  Run validator in dev mode

```bash
task validator
```
