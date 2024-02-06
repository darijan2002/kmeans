const MAX_ITERATIONS = 100;
const NUM_CLUSTERS = 15;
const FILENAME = "image.jpg";

function randomBetween(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}

function calcMeanCentroid(dataSet, start, end) {
  const features = dataSet[0].length;
  const n = end - start;
  let mean = [];
  for (let i = 0; i < features; i++) {
    mean.push(0);
  }
  for (let i = start; i < end; i++) {
    for (let j = 0; j < features; j++) {
      mean[j] = mean[j] + dataSet[i][j] / n;
    }
  }
  return mean;
}

function getRandomCentroidsNaiveSharding(dataset, k) {
  // implementation of a variation of naive sharding centroid initialization method
  // (not using sums or sorting, just dividing into k shards and calc mean)
  // https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
  const numSamples = dataset.length;
  // Divide dataset into k shards:
  const step = Math.floor(numSamples / k);
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const start = step * i;
    let end = step * (i + 1);
    if (i + 1 === k) {
      end = numSamples;
    }
    centroids.push(calcMeanCentroid(dataset, start, end));
  }
  return centroids;
}

function getRandomCentroids(dataset, k) {
  // selects random points as centroids from the dataset
  const numSamples = dataset.length;
  const centroidsIndex = [];
  let index;
  while (centroidsIndex.length < k) {
    index = randomBetween(0, numSamples);
    if (centroidsIndex.indexOf(index) === -1) {
      centroidsIndex.push(index);
    }
  }
  const centroids = [];
  for (let i = 0; i < centroidsIndex.length; i++) {
    const centroid = [...dataset[centroidsIndex[i]]];
    centroids.push(centroid);
  }
  return centroids;
}

function compareCentroids(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function shouldStop(oldCentroids, centroids, iterations) {
  if (iterations > MAX_ITERATIONS) {
    return true;
  }
  if (!oldCentroids || !oldCentroids.length) {
    return false;
  }
  let sameCount = true;
  for (let i = 0; i < centroids.length; i++) {
    if (!compareCentroids(centroids[i], oldCentroids[i])) {
      sameCount = false;
    }
  }
  return sameCount;
}

// Calculate Squared Euclidean Distance
function getDistanceSQ(a, b) {
  const diffs = [];
  for (let i = 0; i < a.length; i++) {
    diffs.push(a[i] - b[i]);
  }
  return diffs.reduce((r, e) => r + e * e, 0);
}

// Returns a label for each piece of data in the dataset.
function getLabels(dataSet, centroids) {
  // prep data structure:
  const labels = {};
  for (let c = 0; c < centroids.length; c++) {
    labels[c] = {
      points: [],
      centroid: centroids[c],
    };
  }
  // For each element in the dataset, choose the closest centroid.
  // Make that centroid the element's label.
  for (let i = 0; i < dataSet.length; i++) {
    const a = dataSet[i];
    let closestCentroid, closestCentroidIndex, prevDistance;
    for (let j = 0; j < centroids.length; j++) {
      let centroid = centroids[j];
      if (j === 0) {
        closestCentroid = centroid;
        closestCentroidIndex = j;
        prevDistance = getDistanceSQ(a, closestCentroid);
      } else {
        // get distance:
        const distance = getDistanceSQ(a, centroid);
        if (distance < prevDistance) {
          prevDistance = distance;
          closestCentroid = centroid;
          closestCentroidIndex = j;
        }
      }
    }
    // add point to centroid labels:
    labels[closestCentroidIndex].points.push(a);
  }
  return labels;
}

function getPointsMean(pointList) {
  const totalPoints = pointList.length;
  const means = [];
  for (let j = 0; j < pointList[0].length; j++) {
    means.push(0);
  }
  for (let i = 0; i < pointList.length; i++) {
    const point = pointList[i];
    for (let j = 0; j < point.length; j++) {
      const val = point[j];
      means[j] = means[j] + val / totalPoints;
    }
  }
  return means;
}

function recalculateCentroids(dataSet, labels, k) {
  // Each centroid is the geometric mean of the points that
  // have that centroid's label. Important: If a centroid is empty (no points have
  // that centroid's label) you should randomly re-initialize it.
  let newCentroid;
  const newCentroidList = [];
  for (const k in labels) {
    const centroidGroup = labels[k];
    if (centroidGroup.points.length > 0) {
      // find mean:
      newCentroid = getPointsMean(centroidGroup.points);
    } else {
      // get new random centroid
      newCentroid = getRandomCentroids(dataSet, 1)[0];
    }
    newCentroidList.push(newCentroid);
  }
  return newCentroidList;
}

let results = {};
let iterations = 0;
let oldCentroids, labels, centroids, clusters;
let dataset, k;
function doIter(ev) {
  if (ev.code !== "Enter") return;

  if (!shouldStop(oldCentroids, centroids, iterations)) {
    // Save old centroids for convergence test.
    oldCentroids = [...centroids];
    iterations++;

    // Assign labels to each datapoint based on centroids
    labels = getLabels(dataset, centroids);
    centroids = recalculateCentroids(dataset, labels, k);

    canvas1.loop();
    return;
  }

  clusters = [];
  for (let i = 0; i < k; i++) {
    clusters.push(labels[i]);
  }
  results = {
    clusters: clusters,
    centroids: centroids,
    iterations: iterations,
    converged: iterations <= MAX_ITERATIONS,
  };
  console.log(results);

  canvas1.noStroke();
  canvas1.fill(0);
  canvas1.text("generated palette:", 10, 315);
  for (const c in centroids) {
    canvas1.fill(...centroids[c]);
    canvas1.square(10 + c * 20, 320, 20);
  }
  for (let c = 0; c < clusters.length; c++) {
    let s = new Set();
    for (let i = 0; i < clusters[c].points.length; i++) {
      const p = clusters[c].points[i];
      // dataset.push(img.pixels.slice(i, i + 3));
      const r = p[0] << 16;
      const g = p[1] << 8;
      const b = p[2];
      s.add(r | g | b);
    }
    clusters[c].points = s;
  }

  for (let i = 0; i < 4 * img.width * img.height; i += 4) {
    const [r, g, b] = img.pixels.slice(i, i + 3);
    const col = (r << 16) | (g << 8) | b;
    let cl = null;
    for (const c of clusters) {
      if (c.points.has(col)) {
        cl = c.centroid;
        break;
      }
    }
    img.pixels[i] = cl[0];
    img.pixels[i + 1] = cl[1];
    img.pixels[i + 2] = cl[2];
  }
  img.updatePixels();
  {
    const w = canvas1.width / img.width;
    if(w < 1) canvas1.image(img, 0, 400, img.width*w, img.height*w);
    else canvas1.image(img, 0, 400);
  }

  document.body.removeEventListener("keydown", doIter);
}

function kmeans(_dataset, _k, useNaiveSharding = true) {
  // Initialize book keeping variables
  dataset = _dataset;
  k = _k;

  // Initialize centroids randomly
  if (useNaiveSharding) {
    centroids = getRandomCentroidsNaiveSharding(dataset, k);
  } else {
    centroids = getRandomCentroids(dataset, k);
  }

  document.body.addEventListener("keydown", doIter);
  // Run the main k-means algorithm
}

let inputElement;
let img = null;
let rg_dataset, rb_dataset, gb_dataset;
let canvas1 = new p5((s) => {
  s.preload = function () {
    img = s.loadImage(FILENAME);
  };

  s.setup = function () {
    s.createCanvas(1000, 1000);
    s.background('#ccc');
    img.loadPixels();
    // dataset = [];
    dataset = new Set();
    for (let i = 0; i < 4 * img.width * img.height; i += 4) {
      // dataset.push(img.pixels.slice(i, i + 3));
      const r = img.pixels[i] << 16;
      const g = img.pixels[i + 1] << 8;
      const b = img.pixels[i + 2];
      dataset.add(r | g | b);
    }
    dataset = [...dataset];
    for (let i = 0; i < dataset.length; i++) {
      dataset[i] = [
        dataset[i] >> 16,
        (dataset[i] >> 8) & 0xff,
        dataset[i] & 0xff,
      ];
    }
    kmeans(dataset, NUM_CLUSTERS);

    rg_dataset = s.createGraphics(300, 300);
    dataset.forEach((p) => {
      rg_dataset.noStroke();
      rg_dataset.fill(p[0], p[1], p[2]);
      rg_dataset.circle(p[0], p[1], 3);
    });
    rg_dataset.noStroke();
    rg_dataset.fill(0);
    rg_dataset.text("x-axis: red\ny-axis: green", 0, 260);

    rb_dataset = s.createGraphics(300, 300);
    dataset.forEach((p) => {
      rb_dataset.noStroke();
      rb_dataset.fill(p[0], p[1], p[2]);
      rb_dataset.circle(p[0], p[2], 3);
    });
    rb_dataset.noStroke();
    rb_dataset.fill(0);
    rb_dataset.text("x-axis: red\ny-axis: blue", 0, 260);

    gb_dataset = s.createGraphics(300, 300);
    dataset.forEach((p) => {
      gb_dataset.noStroke();
      gb_dataset.fill(p[0], p[1], p[2]);
      gb_dataset.circle(p[1], p[2], 3);
    });
    gb_dataset.noStroke();
    gb_dataset.fill(0);
    gb_dataset.text("x-axis: green\ny-axis: blue", 0, 260);

    {
      const w = s.width / img.width;
      if(w < 1) s.image(img, 0, 400, img.width*w, img.height*w);
      else s.image(img, 0, 400);
    }
    s.print(dataset);
  };

  s.draw = function () {
    s.noStroke();
    s.fill("#ccc");
    s.rect(0,0,1000,400);
    s.fill(0);
    s.text(`iteration ${iterations} out of ${MAX_ITERATIONS} max`, 10, 350);

    s.image(rg_dataset, 10, 10);
    s.image(rb_dataset, 310, 10);
    s.image(gb_dataset, 610, 10);
    centroids.forEach((p) => {
      s.fill(p[0], p[1], p[2]);
      s.stroke(255-p[0], 255-p[1], 255-p[2]);
      s.strokeWeight(1);
      s.square(10 + p[0] - 1, 10 + p[1] - 1, 3);
      s.square(310 + p[0] - 1, 10 + p[2] - 1, 3);
      s.square(610 + p[1] - 1, 10 + p[2] - 1, 3);
    });
    s.noLoop();
  };
}, "canvas1");
