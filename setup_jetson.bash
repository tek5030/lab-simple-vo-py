#!/usr/bin/env bash
setup_jetson() (
  pimp_text() {
    green=$(tput setaf 2)
    bold=$(tput bold)   # Start bold text
    dim=$(tput dim)     # Start half intensity mode
    normal=$(tput sgr0) # Turn off all attributes
  }

  pimp_text
  up() { tput cuu1; }
  down() { tput cud1; }
  check() { up; printf "${dim}[${green}${bold}✓${normal}\r"; down; }

  set -eu
  printf "${green}running setup for jetson.${normal}\n"

  # install librealsense requirements
  ./libuvc_installation.sh

  printf "${dim}[ ] create virtual environment ...${normal}\n"
  python3.8 -m venv venv
  check

  printf "${dim}[ ] ammend PYTHONPATH in venv...${normal}\n"
  echo 'export PYTHONPATH=${PWD}/venv/lib/python3.8/site-packages/pyrealsense2' >> venv/bin/activate
  source venv/bin/activate
  check

  printf "${dim}[ ] download vtk...${normal}\n"
  wget -qi url-vtk.txt -O "vtk-9.0.1-cp38-cp38-linux_aarch64.whl"
  check

  printf "${dim}[ ] download and install pyrealsense2 ...${normal}\n"
  wget -qi url-pyrealsense2.txt -O- | tar xzf - -C venv/lib/python3.8/site-packages/
  check

  printf "${dim}[ ] install requirements ...${normal}\n"
  pip install -q -U pip wheel
  pip install -q -r requirements-jetson.txt
  check

  printf "${green}setup complete!${normal}\n"
)

[[ $_ != $0 ]] || setup_jetson
