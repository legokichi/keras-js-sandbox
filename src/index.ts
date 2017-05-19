const KerasJS = <any>require('../keras-js/dist/keras.js');


async function main(){
  console.log("main");
  const model = new KerasJS.Model({
    filepaths: {
      model: '../model.json',
      weights: '../model_weights.buf',
      metadata: '../model_metadata.json'
    },
    gpu: true
  });
  console.time("ready");
  await model.ready();
  console.timeEnd("ready");
  const img = await new Promise<HTMLImageElement>((resolve, reject)=>{
    const img = new Image();
    img.src = "../crop.jpg";
    img.onload = ()=> resolve(img);
    img.onerror = reject;
  });
  const ctx = <CanvasRenderingContext2D>document.createElement("canvas").getContext("2d");
  ctx.canvas.width = img.width;
  ctx.canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
  const imgdata = ctx.getImageData(0, 0, img.width, img.height);
  if(imgdata.width !== 224 || imgdata.height !== 224 || imgdata.data.length !== (3+1)*224*224){
    console.error(imgdata);
    throw new Error("image size error");
  }
  const data = new Float32Array(3*224*224);
  let ptr = 0;
  for(let i=0; i<imgdata.data.length; i+=4){
    const r = imgdata.data[i+0];
    const g = imgdata.data[i+1];
    const b = imgdata.data[i+2];
    const a = imgdata.data[i+3];
    data[ptr+0] = r;
    data[ptr+1] = g;
    data[ptr+2] = b;
    ptr += 3;
  }
  const inputData = { 'input_1': new Float32Array(data) };
  console.time("predict");
  const outputData = await model.predict(inputData);
  console.timeEnd("predict");
  console.log(outputData);
  let max_val = 0;
  let max_index = 0;
  outputData["predictions"].forEach((val, i)=>{
    if(max_val < val){ max_index = i; }
  });
  console.log("category: ", max_index, "val:", max_val);
}


main();
