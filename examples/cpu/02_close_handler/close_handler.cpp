#include <iostream>
#include <signal.h>

bool forever = true;

void sighandler(int sig) {
  std::cout << "Signal " << sig << " caught..." << std::endl;
  forever = false;
}

int main() {
  signal(SIGABRT, &sighandler);
  signal(SIGTERM, &sighandler);
  signal(SIGINT, &sighandler);

  while (forever) {
  }
}
