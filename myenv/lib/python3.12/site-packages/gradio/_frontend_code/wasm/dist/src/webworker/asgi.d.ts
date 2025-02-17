import type { ASGIApplication, ASGIScope } from "../asgi-types";
export declare function makeAsgiRequest(asgiApp: ASGIApplication, scope: ASGIScope, messagePort: MessagePort): Promise<void>;
