interface Value {
    lines: number;
    max_lines: number;
    text: string;
}
export declare function resize(target: HTMLTextAreaElement | HTMLInputElement, lines: number, max_lines: number): Promise<void>;
export declare function text_area_resize(_el: HTMLTextAreaElement, _value: Value): any | undefined;
export {};
