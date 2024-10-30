import init from './pkg/eversion.js';

console.log("Loaded");

async function run() {
    try {
        await init();
    } catch(e) {
        console.error(e);
    }
}

run();