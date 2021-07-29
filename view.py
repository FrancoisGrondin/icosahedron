import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():

	parser = argparse.ArgumentParser(description='Visualize sphere.')
	parser.add_argument('--file', type=str, default="", help='Npy file name to load points')
	args = parser.parse_args()

	points = np.load(args.file)

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(points[:,0], points[:,1], points[:,2])
	plt.show()

if __name__ == "__main__":

	main()