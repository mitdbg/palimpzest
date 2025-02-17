import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.js";
declare const __propDef: {
    props: {
        record: RecordPlugin;
        i18n: I18nFormatter;
        recording?: boolean | undefined;
        record_time: string;
        show_recording_waveform: boolean | undefined;
        timing?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type WaveformRecordControlsProps = typeof __propDef.props;
export type WaveformRecordControlsEvents = typeof __propDef.events;
export type WaveformRecordControlsSlots = typeof __propDef.slots;
export default class WaveformRecordControls extends SvelteComponent<WaveformRecordControlsProps, WaveformRecordControlsEvents, WaveformRecordControlsSlots> {
}
export {};
