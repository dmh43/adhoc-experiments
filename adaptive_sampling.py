import numpy as np
import matplotlib.pyplot as plt

def main():
  arm_1 = []
  arm_2 = []
  for i in range(10000):
    arm_1_first = np.random.normal(size=50)
    arm_2_first = np.random.normal(size=50)
    if arm_1_first.mean() > arm_2_first.mean():
      arm_1_second = np.random.normal(size=90)
      arm_2_second = np.random.normal(size=10)
    else:
      arm_1_second = np.random.normal(size=10)
      arm_2_second = np.random.normal(size=90)
    arm_1.append(np.r_[arm_1_first, arm_1_second].mean())
    arm_2.append(np.r_[arm_2_first, arm_2_second].mean())
  plt.hist(arm_1, alpha=0.7, label='arm1', bins=30)
  plt.hist(arm_2, alpha=0.7, label='arm2', bins=30)
  plt.legend()
  plt.show()


def main():
  arm_1 = []
  arm_2 = []
  for i in range(10000):
    arm_1_first = np.random.normal(size=50)
    arm_2_first = np.random.normal(size=50)
    if arm_1_first.mean() > arm_2_first.mean():
      arm_1_second = np.random.normal(size=90)
      arm_2_second = np.random.normal(size=10)
      arm_1.append(np.r_[arm_1_first, arm_1_second].mean())
      arm_2.append(np.r_[arm_2_first, arm_2_second].mean())
    else:
      arm_1_second = np.random.normal(size=10)
      arm_2_second = np.random.normal(size=90)
      arm_2.append(np.r_[arm_1_first, arm_1_second].mean())
      arm_1.append(np.r_[arm_2_first, arm_2_second].mean())
  plt.hist(arm_1, alpha=0.7, label='had larger initial mean', bins=30)
  plt.hist(arm_2, alpha=0.7, label='had smaller initial mean', bins=30)
  plt.legend()
  plt.show()


if __name__ == '__main__': main()
