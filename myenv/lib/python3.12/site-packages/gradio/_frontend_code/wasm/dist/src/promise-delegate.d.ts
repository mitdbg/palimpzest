type PromiseImplFn<T> = ConstructorParameters<typeof Promise<T>>[0];
export declare class PromiseDelegate<T> {
    promiseInternal: Promise<T>;
    resolveInternal: Parameters<PromiseImplFn<T>>[0];
    rejectInternal: Parameters<PromiseImplFn<T>>[1];
    constructor();
    get promise(): Promise<T>;
    resolve(value: T): void;
    reject(reason: unknown): void;
}
export {};
