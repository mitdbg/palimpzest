export declare function handle_filter(choices: [string, string | number][], input_text: string): number[];
export declare function handle_change(dispatch: any, value: string | number | (string | number)[] | undefined, value_is_output: boolean): void;
export declare function handle_shared_keys(e: KeyboardEvent, active_index: number | null, filtered_indices: number[]): [boolean, number | null];
