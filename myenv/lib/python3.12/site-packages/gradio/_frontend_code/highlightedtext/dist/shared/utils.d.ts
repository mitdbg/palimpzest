export declare function name_to_rgba(name: string, a: number, ctx: CanvasRenderingContext2D | null): string;
export declare function correct_color_map(color_map: Record<string, string>, _color_map: Record<string, {
    primary: string;
    secondary: string;
}>, browser: any, ctx: CanvasRenderingContext2D | null): void;
export declare function merge_elements(value: {
    token: string;
    class_or_confidence: string | number | null;
}[], mergeMode: "empty" | "equal"): {
    token: string;
    class_or_confidence: string | number | null;
}[];
