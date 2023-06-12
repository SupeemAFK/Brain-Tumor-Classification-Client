import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { AiOutlineUpload } from 'react-icons/ai'
import { CircularProgressbarWithChildren } from 'react-circular-progressbar';

import 'react-circular-progressbar/dist/styles.css';

interface IPredictImage {
  imageId: number;
  percent: number;
}

interface IImage {
  id: number;
  src: string;
}

function App() {
  const [images, setImages] = useState<IImage[]>([]);
  const [tfModel, setTfModel] = useState<tf.LayersModel>();
  const [predictedValue, setPredictedValue] = useState<IPredictImage[]>([]);

  useEffect(() => {
    async function loadModel() {
      const model = await tf.loadLayersModel('../tsflw/tfjsmodel/model.json');
      setTfModel(model);
    }
    loadModel()
  }, [])

  useEffect(() => {
    if (images.length > 0) {
      images.forEach(image => {
        const im = new Image();
        im.width = 256
        im.height = 256
        im.src = image.src
        im.onload = () => {
          const tensor = convertToTensor(im);
          const predicted = predict(tf.expandDims(tensor, 0))
          const predictImage: IPredictImage = { imageId: image.id, percent: predicted }
          setPredictedValue(prevPredictedValue => [...prevPredictedValue, predictImage])
        }
      })
    }
  }, [images])

  function predict(input: tf.Tensor<tf.Rank>) {
    const tensor = tfModel?.predict(input) as tf.Tensor
    const value = tensor.dataSync()[0]
    return value * 100
  }

  function convertToTensor(imgRef: HTMLImageElement) {
    const a = tf.browser.fromPixels(imgRef).resizeBilinear([256,256])
    const normalized = a.div(tf.scalar(255))
    return normalized
  }

  function handleOnChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files) return;
    setPredictedValue([])
    setImages([])
    const images = e.target.files;
    const url = Array.from(images).map((img: File, i) => ({ id: i, src: URL.createObjectURL(img) }))
    setImages(url)
  }

  return (
    <div className='flex flex-col justify-center items-center h-screen'>
      <div className='flex-[0.2] flex-col flex justify-center items-center w-full'>
        <label className={`${images.length == 0 ? "upload-btn-main" : "upload-btn-secondary"}`}>
          {images.length == 0 ? "อัพโหลดรูปภาพ" : "อัพโหลดรูปภาพใหม่"}
          <AiOutlineUpload className='text-3xl mt-1' />
          <input onChange={handleOnChange} type="file" className="hidden" multiple />
        </label>
        <p className='text-xs text-red-500 mt-1 px-5'>*โปรดเลือกรูปภาพ MRI ที่มีความคมชัด เพื่อความถูกต้องในการประมวณผลของ AI</p>
      </div>
      {images.length > 0 && (
        <div className='flex-[0.8] flex justify-center items-center overflow-y-auto flex-wrap w-full p-5'>
          {images.map(img => (
            <div key={img.id} className='mx-1 mt-3 flex flex-col items-center shadow-md border-[1px] border-[#e6e6e6] rounded-lg p-3'>
              <div>
                <img className='w-52 h-52 rounded-md' src={img.src} alt="image" />
              </div>

              <div className='w-44 h-44 mt-2'>
                <CircularProgressbarWithChildren 
                  maxValue={100}
                  styles={{ path: { stroke: "#3b82f6" }}}
                  value={predictedValue?.find(value => value.imageId == img.id)?.percent ? Number(predictedValue.find(value => value.imageId == img.id)?.percent.toFixed(2)) : 0 } 
                >
                  <div className="flex flex-col items-center">
                    <strong className="font-light">{predictedValue?.find(value => value.imageId == img.id)?.percent ? "AI ประมวลผลได้" : "กำลังประมวลผล"}</strong>
                    <strong className="text-3xl">{predictedValue?.find(value => value.imageId == img.id)?.percent && `${predictedValue.find(value => value.imageId == img.id)?.percent.toFixed(2)} %`}</strong>
                  </div>
                </CircularProgressbarWithChildren>
              </div>

              <div className='mt-1'>
                {predictedValue?.find(value => value.imageId == img.id) ? (
                  <div>
                    <div className='mt-2'>
                      { 
                        Number(predictedValue?.find(value => value.imageId == img.id)?.percent) >= 20 ? 
                          Number(predictedValue?.find(value => value.imageId == img.id)?.percent) >= 50 ? 
                            <p className='text-red-500 font-semibold text-center'>มีโอกาสเป็นเนื้องอกในสมอง!</p>
                            : 
                            <p className='text-yellow-500 font-semibold text-center'>มีโอกาสเป็นเนื้องอกในสมอง</p> 
                          : 
                          <p className='text-blue-500 font-semibold text-center'>ไม่พบเนื้องอกในสมอง</p>
                      }
                    </div>
                  </div>
                ) : (
                  <div>
                    <p className='text-center'>กำลังประมวลผลด้วย AI...</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
    
  )
}

export default App
