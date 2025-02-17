import { AwaitableQueue } from "./awaitable-queue";
// Connect the `messagePort` to the `asgiApp` so that
// the `asgiApp` can receive ASGI events (`ReceiveEvent`) from the `messagePort`
// and send ASGI events (`SendEvent`) to the `messagePort`.
export function makeAsgiRequest(asgiApp, scope, messagePort) {
    const receiveEventQueue = new AwaitableQueue();
    messagePort.addEventListener("message", (event) => {
        receiveEventQueue.enqueue(event.data);
    });
    messagePort.start();
    // Set up the ASGI application, passing it the `scope` and the `receive` and `send` functions.
    // Ref: https://asgi.readthedocs.io/en/latest/specs/main.html#applications
    async function receiveFromJs() {
        return await receiveEventQueue.dequeue();
    }
    async function sendToJs(proxiedEvent) {
        const event = Object.fromEntries(proxiedEvent.toJs());
        messagePort.postMessage(event);
    }
    return asgiApp(scope, receiveFromJs, sendToJs);
}
