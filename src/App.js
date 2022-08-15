import './App.css';
import React from 'react';
import Webcam from 'react-webcam';

// const ort = require('onnxruntime-web');
/*global ort */

const vocab = ["book", "plant"];

const SQSZ = 224 // this square size is everywhere

const isRorGorB = (bchw1x3xSQSZxSQSZ) => {
  // do the dumbest thing and average each color plane, and softmax the results, all in Javascript.
  // this is a shim for an onnx model obviously
  const rgbsums = [0,1,2].map(c => {
    let sum = 0.0;
    for(let i=c*SQSZ*SQSZ; i<(c+1)*SQSZ*SQSZ; i++) {
      sum += bchw1x3xSQSZxSQSZ[i]
    }
    return sum / (SQSZ * SQSZ);
  })

  const softmax = Array.from({length: rgbsums.length}, (_,i) => (Math.exp(rgbsums[i] * 10) / rgbsums.reduce((z,e) => z + Math.exp(e * 10), 0)))
  return softmax;
}

const unclampAndTranspose201 = (bhwc, bchw) => {
  // 8-bit bhwc canvas ImageData -> float32 bchw
  for(let c=0;c<=2;c++) {
    for(let h=0;h<SQSZ;h++) {
      for(let w=0;w<SQSZ;w++) {
        bchw[c*SQSZ*SQSZ + h*SQSZ + w] = bhwc[4 * (h * SQSZ + w) + c] / 255.0
      }
    }
  }
}

const videoConstraints = {
  width: SQSZ,
  height: SQSZ,
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      bpstats: [0.1,0.1],
      timing: "",
      bchw: Float32Array.from({length: SQSZ * SQSZ * 3}),
      facingMode: "environment",
    };
    this.webcamref = React.createRef();
  }
  componentDidMount() {
    this.handleWebcam();
    this.handleOnnx();
  }
  async handleOnnx() {
    this.setState({ bpstats: [0.2, 0.2]})
    const ortsession = await ort.InferenceSession.create('./model.onnx');
    this.setState({ bpstats: [0.3, 0.3]})
    this.setState({ ortsession })
    this.setState({ bpstats: [0.4, 0.4]})
  }
  handleWebcam() {
    if(this.webcamref && this.webcamref.current && this.webcamref.current.getCanvas) {
      const canvas = this.webcamref.current.getCanvas();
      if (canvas !== null && this.state.ortsession !== undefined) {
        const ctx = canvas.getContext("2d");
        if (ctx !== null) {
          const imagedata = ctx.getImageData(0, 0, SQSZ, SQSZ).data;
          this.ingestPicture(imagedata);
        } else {
          console.log("no 2d context lol");
          requestAnimationFrame(() => this.handleWebcam());
        }
      } else {
        this.setState({timing: `canvas ${canvas !== null} session ${this.state.ortsession !== undefined}`});
        requestAnimationFrame(() => this.handleWebcam());
      }
    }
  }
  async ingestPicture(imagedata) {
    const startDate = Date.now();
    const {bchw, ortsession} = this.state;
    unclampAndTranspose201(imagedata, bchw);
    const xposeDate = Date.now();
    const input = new ort.Tensor("float32", bchw, [1, 3, SQSZ, SQSZ]);
    const {output} = await ortsession.run({input});
    const bpstats = output.data;
    const inferDate = Date.now();
    const timing = `${xposeDate - startDate} / ${inferDate - startDate}`;
    this.setState({bpstats, timing});
    requestAnimationFrame(() => this.handleWebcam());
  }
  render() {
    const {bpstats: [bstat, pstat], timing} = this.state;
    return <div className="App">
      <h1>
        <span style={{opacity: bstat}} >BOOK</span>
        &nbsp;
        <span style={{opacity: pstat}} >PLANT</span>
      </h1>
      <h2>
        via your&nbsp;
        <select id="facingMode" onChange={e => this.setState({facingMode: e.target.value})} value={this.state.facingMode}>
          <option value="user">selfie</option>
          <option value="environment">photo</option>
          <option value=""></option>
        </select>&nbsp;webcam
      </h2>
      <Webcam
        audio={false}
        height={videoConstraints.height}
        width={videoConstraints.width}
        ref={this.webcamref}
        videoConstraints={{...videoConstraints, facingMode: this.state.facingMode}}
      />
      <div className="millis">
        {timing} ms: {bstat.toFixed(3)} book, {pstat.toFixed(3)} plant {this.state.ortsession === null ? " --- no session :(" : ""}
      </div>
    </div>
  }
}

export default App;
