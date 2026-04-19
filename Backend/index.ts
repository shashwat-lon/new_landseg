import express from 'express';
import axios from 'axios';
import * as ort from 'onnxruntime-node';
import * as Jimp from 'jimp';
import { createCanvas } from '@napi-rs/canvas';
import fs from 'fs';
import cors from 'cors';
import dotenv from 'dotenv';
import * as tilebelt from '@mapbox/tilebelt';


dotenv.config();

const app = express();
app.use(cors());

const loadImageFromURL = async (url: string, path: string): Promise<void> => {
  const response = await axios.get(url, { responseType: 'arraybuffer' });
  fs.writeFileSync(path, Buffer.from(response.data));
};

const imageDataToTensor = async (image: Jimp, dims: number[]) => {
  const imageBufferData = image.bitmap.data;
  const image_arr: number[] = [];

  for (let i = 0; i < imageBufferData.length; i += 4) {
    image_arr.push(imageBufferData[i]);     // R
    image_arr.push(imageBufferData[i + 1]); // G
    image_arr.push(imageBufferData[i + 2]); // B
  }

  if (image_arr.length !== dims[1] * dims[2] * dims[3]) {
    throw new Error(
      `Tensor data length mismatch: got ${image_arr.length}, expected ${dims[1] * dims[2] * dims[3]}`
    );
  }

  const floatImage = Float32Array.from(image_arr.map((val) => val / 255.0));
  return new ort.Tensor('float32', floatImage, dims);
};

const getImageTensorFromPath = async (path: string, dims: number[] = [1, 256, 256, 3]) => {
  const image = await Jimp.read(path);

  if (image.bitmap.width !== 256 || image.bitmap.height !== 256) {
    image.resize(256, 256);
  }

  return imageDataToTensor(image, dims);
};

const runInference = async (session: ort.InferenceSession, preprocessedData: ort.Tensor): Promise<[number[][], number]> => {
  const start = Date.now();
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;

  const outputData = await session.run(feeds);
  const inferenceTime = (Date.now() - start) / 1000;
  const output = outputData[session.outputNames[0]];
  const outputArray = Array.from(output.data as Iterable<number>);

  const num_class = 5;
  const prediction: number[][] = [];

  for (let i = 0; i < outputArray.length; i += num_class * 256) {
    const row = [];
    for (let j = 0; j < 256 * num_class; j += num_class) {
      const scores = outputArray.slice(i + j, i + j + num_class);
      const maxIndex = scores.indexOf(Math.max(...scores));
      row.push(maxIndex);
    }
    prediction.push(row);
  }

  return [prediction, inferenceTime];
};

const ort_run_inference = async () => {
  const session = await ort.InferenceSession.create('model/unet_poland_ds_modelv1.onnx');
  const path = 'queried_image.png';
  const imageTensor = await getImageTensorFromPath(path);
  return runInference(session, imageTensor);
};

app.get('/query_segment', async (req, res) => {
  try {
    const { latitude, longitude } = req.query;
    const lat = parseFloat(latitude as string) || 52.237;
    const lng = parseFloat(longitude as string) || 21.017;
    const zoom = 18;

    const [xTile, yTile] = tilebelt.pointToTile(lng, lat, zoom);
    const tileUrl = `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${yTile}/${xTile}`;
    const imagePath = 'queried_image.png';

    console.log(`Fetching tile from: ${tileUrl}`);
    await loadImageFromURL(tileUrl, imagePath);

    const [prediction, inferenceTime] = await ort_run_inference();

    const canvasNode = createCanvas(256, 256);
    const ctx = canvasNode.getContext("2d");
    const colours = ["#4A235A", "#FFB900", "#0FFF00", "#008BFF", "#FFF600"];
    const class_count = [0, 0, 0, 0, 0];

    for (let row = 0; row < 256; row++) {
      for (let col = 0; col < 256; col++) {
        const label = prediction[row][col];
        ctx.fillStyle = colours[label];
        ctx.fillRect(col, row, 1, 1);
        class_count[label]++;
      }
    }

    const baseImage = fs.readFileSync(imagePath);
    const buffer = canvasNode.toBuffer('image/png');

    res.json({
      ori_img: baseImage.toString('base64'),
      pred_result: buffer.toString('base64'),
      class_count,
      inferenceTime
    });
  } catch (error) {
    console.error('Error in /query_segment:', error);
    res.status(500).send('Something went wrong');
  }
});

app.listen(process.env.PORT || 3001, () => {
  console.log(`⚡️[server]: Running on port ${process.env.PORT || 3001}`);
});
