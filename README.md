# Color palette extraction with K-means clustering

# Demo
https://github.com/darijan2002/kmeans/assets/12586177/b8c2b1fc-1736-4218-9b8a-bcc653eac8dc

# Usage
When you open the page, there will be three plots for each projection of the 
3D color space (red-green plane, red-blue plane, green-blue plane). To run an 
iteration press the `Enter` key. When the algorithm converges (or the max 
number of iterations is passed), the page will display the generated palette of
the most representative colors of the image and repaint the image using
those colors.

To run the code, start a local server in this directory
e.g. with Python

```bash
python3 -m http.server
```

and navigate to the `index.html` file

# Parameters

- To change the image used in the example, change the `FILENAME` constant
- To change the max number of iterations change the `MAX_ITERATIONS` constant
- To change the number of clusters change the `NUM_CLUSTERS` constant
