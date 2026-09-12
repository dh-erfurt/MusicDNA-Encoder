// Read {name: pae} on stdin, write {name: mei-or-null} on stdout.
//
// Verovio is RISM Digital's own renderer and the reference implementation of
// Plaine & Easie. Its published Python wheels are built with NO_PAE_SUPPORT, so
// the reader is only reachable through the JavaScript/WASM toolkit -- which is
// why a Python package carries a Node helper. See tests/test_plaine_easie_roundtrip.py.

const vrv = require("verovio");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  vrv.module.onRuntimeInitialized = () => {
    const incipits = JSON.parse(input);
    const rendered = {};
    for (const [name, pae] of Object.entries(incipits)) {
      const toolkit = new vrv.toolkit();
      rendered[name] = toolkit.loadData(pae) ? toolkit.getMEI() : null;
    }
    process.stdout.write(JSON.stringify(rendered));
  };
});
