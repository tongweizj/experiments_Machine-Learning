// this example converts Die_Live integer encoding of categorical class data
// to one-hot encoding
const arrayData = [{"Die_Live": 1}, {"Die_Live": 1}, {"Die_Live": 2}, 
                    {"Die_Live": 1}, {"Die_Live": 2 }];

console.log (arrayData);

// assume that arrayData contans training data (array of json objects)
const outputData = arrayData.map(item => [
    item.Die_Live === 1 ? 1 : 0,
    item.Die_Live === 2 ? 1 : 0
])
//
console.log (outputData);
