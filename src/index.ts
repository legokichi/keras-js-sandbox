
async function main(){
  console.log("main");
  
  console.time("class_index");
  const class_index: [string, string][] = await fetch("./imagenet_class_index.json").then((res)=> res.json());
  //console.log(class_index);
  console.timeEnd("class_index");
  
  console.time("ready");
  const model = new window["KerasJS"].Model({
    filepaths: {
      model: './model.json',
      weights: './model_weights.buf',
      metadata: './model_metadata.json'
    },
    gpu: true
  });
  await model.ready();

  const [width, height, ch] = model.inputTensors.input_1.tensor.shape;

  //console.log(model);
  console.timeEnd("ready");
  
  const source = await (async ()=>{
    if(1){
      const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
      const url = URL.createObjectURL(stream);
      const video = document.createElement("video");
      video.src = url;
      await new Promise((resolve, reject)=>{
        video.addEventListener("loadeddata", resolve, <any>{once: true});
        video.addEventListener("error", reject, <any>{once: true});
      });
      video.autoplay = true;
      video.width = video.videoWidth;
      video.height = video.videoHeight;
      return video;
    }else{
      const img = await new Promise<HTMLImageElement>((resolve, reject)=>{
        const img = new Image();
        img.src = "../crop.jpg";
        img.onload = ()=> resolve(img);
        img.onerror = reject;
      });
      return img;
    }
  })();

  const ctx = <CanvasRenderingContext2D>(<HTMLCanvasElement>document.getElementById("cnv")).getContext("2d");
  ctx.canvas.width = width;
  ctx.canvas.height = height;

  while(true){
    ctx.drawImage(source, 0, 0, source.width, source.height, 0, 0, width, height);
    const imgdata = ctx.getImageData(0, 0, width, height);
    if(imgdata.data.length !== (ch+1)*width*height){
      console.error(imgdata);
      throw new Error("image size error");
    }

    const data = new Float32Array(ch*width*height);
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

    await new Promise((resolve)=> setTimeout(resolve, 10));
    
    console.time("predict");
    const inputData = { 'input_1': new Float32Array(data) };
    const outputData = await model.predict(inputData);
    console.timeEnd("predict");
    //console.log(outputData);

    await new Promise((resolve)=> setTimeout(resolve, 10));

    const top3: {val: number; index: number}[] = [];
    const output: Float32Array = outputData[Object.keys(outputData)[0]];
    console.time("sort");
    const sorted = new Float32Array(output).sort().reverse();
    console.timeEnd("sort");

    const top10 = Array.from(sorted.slice(0, 10)).map((val, i)=> ({cat: class_index[output.indexOf(sorted[i])], val}));
    const ul = document.getElementById("result");
    if(ul == null){ throw new Error("dom not found"); }
    ul.innerHTML = "";
    console.log(top10);
    top10.forEach(({cat, val})=>{
      const li = document.createElement("li");
      li.appendChild(document.createTextNode(cat[1]));
      ul.appendChild(li);
    });

    await new Promise((resolve)=> setTimeout(resolve, 10));
  }
}
window.addEventListener("error", (ev)=>{
  console.error(ev, ev.error);
  alert(ev+":"+ev.error);
});

main().catch((err)=>{
  console.error(ev, ev.error);
  alert(ev+":"+ev.error);
});
