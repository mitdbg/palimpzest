export class PromiseDelegate {
    promiseInternal;
    resolveInternal;
    rejectInternal;
    constructor() {
        this.promiseInternal = new Promise((resolve, reject) => {
            this.resolveInternal = resolve;
            this.rejectInternal = reject;
        });
    }
    get promise() {
        return this.promiseInternal;
    }
    resolve(value) {
        this.resolveInternal(value);
    }
    reject(reason) {
        this.rejectInternal(reason);
    }
}
