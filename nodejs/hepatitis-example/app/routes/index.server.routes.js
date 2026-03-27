// Load the 'index' controller
const index = require('../controllers/index.server.controller');


// Define the routes module' method
module.exports = function (app) {

    app.get('/', function (req, res) {
        // render the index.ejs page
        res.render('index.ejs');
    });

    app.get('/train_model', index.trainAndPredict);

};