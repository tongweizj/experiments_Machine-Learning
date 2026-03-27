exports.trainAndPredict = function (req, res) {
    const tf = require('@tensorflow/tfjs');
    //require('@tensorflow/tfjs-node');

    // Load hepatitis dataset
    const hep = require('../../hep.json');
    const hepTesting = require('../../hep_test.json');

    // Process training data
    let trainingData = tf.tensor2d(hep.map(item => [
        item.Age, item.Sex, item.Steroid, item.Antivirals, item.Fatigue, item.Malaise,
        item.Anorexia, item.Liver_big, item.Liver_firm, item.Spleen_palpable,
        item.Spiders, item.Ascites, item.Varices, item.Bilurubin, item.Alk_phosphate,
        item.Sgot, item.Albumin, item.Protime, item.Histology
    ]));

    // Normalize training data
    const { mean, variance } = tf.moments(trainingData, 0);
    const std = variance.sqrt();
    trainingData = trainingData.sub(mean).div(std);

    // Output for training data (adjusted for binary classification)
    const outputData = tf.tensor2d(hep.map(item => [item.Die_Live === 1 ? 0 : 1]));

    // Process testing data similarly
    let testingData = tf.tensor2d(hepTesting.map(item => [
        item.Age, item.Sex, item.Steroid, item.Antivirals, item.Fatigue, item.Malaise,
        item.Anorexia, item.Liver_big, item.Liver_firm, item.Spleen_palpable,
        item.Spiders, item.Ascites, item.Varices, item.Bilurubin, item.Alk_phosphate,
        item.Sgot, item.Albumin, item.Protime, item.Histology
    ]));

    // Normalize testing data
    testingData = testingData.sub(mean).div(std);

    // Build a simpler neural network model
    // Build a simpler neural network model
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [19],
        units: 32,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({l2: 0.01}) // Corrected regularization parameter
    }));
    model.add(tf.layers.dropout(0.5));
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({l2: 0.01}) // Corrected regularization parameter
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    })); // For binary output

    // Compile the model with binary loss function and optimizer
    model.compile({
        loss: 'binaryCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['accuracy'],
    });

    // Train the model
    async function run() {
        await model.fit(trainingData, outputData, {
            epochs: 500, // Adjust epoch count as needed
            validationSplit: 0.2,
            callbacks: [
                // Custom logging callback
                {
                    onEpochEnd: async (epoch, logs) => {
                        console.log(`Epoch ${epoch}: loss = ${logs.loss}, val_loss = ${logs.val_loss}`);
                    }
                },
                // EarlyStopping callback
                //tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 50 })
            ]
        });

        // Predict on testing data
        const predictions = model.predict(testingData);

        // For debugging, explicitly log each prediction
        predictions.array().then(array => {
            console.log(`Total predictions: ${array.length}`);
            array.forEach((prediction, index) => {
                console.log(`Prediction for test instance ${index + 1}: ${prediction}`);
            });

            // Send predictions back in response
            res.json({ predictions: array });
        }).catch(err => {
            console.error("Error processing predictions:", err);
            res.status(500).send('Error processing predictions');
        });

    }

    run().catch(err => res.status(500).send(err));
};
