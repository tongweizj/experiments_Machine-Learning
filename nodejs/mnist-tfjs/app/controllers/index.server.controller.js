

exports.trainAndPredict = function (req, res) {
    /**
     * @license
     * Copyright 2018 Google LLC. All Rights Reserved.
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     * http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     * =============================================================================
     */
    //
    //const tf = require('@tensorflow/tfjs');
    const tf = require('@tensorflow/tfjs-node');
    const argparse = require('argparse');
    //
    const data = require('./data');
    var model = require('./model');
    var preds;
    //
    // run function is our main function
    async function run(epochs, batchSize, modelSavePath) {
        console.log('modelSavePath: ', modelSavePath)
        //load the data
        await data.loadData();
        //
        const { images: trainImages, labels: trainLabels } = data.getTrainData();
        model.summary();

        let epochBeginTime;
        let millisPerStep;
        const validationSplit = 0.15;
        const numTrainExamplesPerEpoch =
            trainImages.shape[0] * (1 - validationSplit);
        console.log("numTrainExamplesPerEpoch",numTrainExamplesPerEpoch) 
   
        const numTrainBatchesPerEpoch =
            Math.ceil(numTrainExamplesPerEpoch / batchSize);
        console.log("numTrainBatchesPerEpoch",numTrainBatchesPerEpoch) 
        //        
        //train the model
        await model.fit(trainImages, trainLabels, {
            epochs,
            batchSize,
            validationSplit
        });
        
        //evaluate the model
        const { images: testImages, labels: testLabels } = data.getTestData();
        console.log("test images length: ", testImages.shape[0]);
        //
        const evalOutput = model.evaluate(testImages, testLabels);
        console.log(
            `\nEvaluation result:\n` +
            `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
            `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);     
        //
        preds = model.predict(testImages).argMax(-1);
        console.log('Predictions after calling predict: ', preds)
        console.log('shape[0]: ', preds.shape[0])
        // save the model
        const saveResults = await model.save('file://./models');
        console.log(`Saved model to path: ./models`);
        //        

    } //end of run function
    //
    // create argument parser to parse command-line arguments
    const parser = new argparse.ArgumentParser({
        description: 'TensorFlow.js-Node MNIST Example.',
        addHelp: true
    });
    // ArgumentParser objects associate
    // command - line arguments with actions.
    //
    //Number of epochs to train the model for 
    parser.addArgument('--epochs', {
        type: 'int',
        defaultValue: 10,
        help: 'Number of epochs to train the model for.'
    });
    //Batch size to be used during model training
    parser.addArgument('--batch_size', {
        type: 'int',
        defaultValue: 128,
        help: 'Batch size to be used during model training.'
    })
    //Path to which the model will be saved after training
    parser.addArgument('--model_save_path', {
        type: 'string',
        defaultValue: './models',
        help: 'Path to which the model will be saved after training.'
    });
    //parse the arguments from the command line
    const args = parser.parseArgs();
    //run with command-line arguments
    //run(args.epochs, args.batch_size, args.model_save_path)
    //run without model_save_path
    run(args.epochs, args.batch_size, args.model_save_path).then((result) => {
        console.log('result:',result)
        console.log ('Predictions: ', preds);
        //res.render('results', { prediction: preds } );
        //
        // get the values from the tf.Tensor
        //var tensorData = results.dataSync();
        // 
        preds.array().then(array => {
            console.log('Array of preds: ',array)
            var resultForTest1 = array[0];
            console.log('resultForTest1: ', resultForTest1)
            var resultForTest2 = array[1];
            var resultForTest3 = array[2];
            //display results in ejs page            
            res.render('results',
                {
                    results: preds,
                    resultForTest1: resultForTest1,
                    resultForTest2: resultForTest2,
                    resultForTest3: resultForTest3
                }
            )
        })



    }).catch((error) => console.log('error:',error));
 
    //

};

//load the model and predict
exports.loadModelAndPredict = function (req, res) {
    //
    const tf = require('@tensorflow/tfjs');
    //
    const data = require('./data');
    var model = require('./model');
    var preds;

    async function run(){
        //load the data
        await data.loadData();
        // 
        const { images: testImages, labels: testLabels } = data.getTestData();
        console.log("test images length: ", testImages.shape[0]);
        // 
        const model = await tf.loadLayersModel('file://./models/model.json');
        //
        preds = model.predict(testImages).argMax(-1);
        console.log('Predictions after calling predict: ', preds)
        console.log('shape[0]: ', preds.shape[0])    }
        //
        run().then((result) => {
            
            console.log('result:',result)
            console.log ('Predictions: ', preds);
            //res.render('results', { prediction: preds } );
            //
            // get the values from the tf.Tensor
            //var tensorData = results.dataSync();
            preds.array().then(array => {
                console.log('Array of preds: ',array)
                var resultForTest1 = array[0];
                console.log('resultForTest1: ', resultForTest1)
                var resultForTest2 = array[1];
                var resultForTest3 = array[2];
                //display results in ejs page            
                res.render('resultsFromModel',
                {
                    results: preds,
                    resultForTest1: resultForTest1,
                    resultForTest2: resultForTest2,
                    resultForTest3: resultForTest3
                })
            
            })
            
        }).catch((error) => console.log('error:',error));
};

