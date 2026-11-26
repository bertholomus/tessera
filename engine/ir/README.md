# engine/ir

<!-- crate table -->
<table>
<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>
<tr><td><a href="graph">graph/</a></td><td>tessera-graph</td><td>The model graph: typed nodes, symbolic shapes, a JSON form</td></tr>
<tr><td><a href="interp">interp/</a></td><td>tessera-interp</td><td>The CPU reference for every plan: runs a graph op by op</td></tr>
<tr><td><a href="kv">kv/</a></td><td>tessera-kv</td><td>Pages of kept rows, shared between runs by refcount</td></tr>
<tr><td><a href="mem">mem/</a></td><td>tessera-mem</td><td>Buffers in zones, typed views onto them, and storage forms</td></tr>
<tr><td><a href="testkit">testkit/</a></td><td>tessera-testkit</td><td>Helpers that tests share: noise, comparisons and scratch files</td></tr>
<tr><td><a href="trace">trace/</a></td><td>tessera-trace</td><td>Builds graphs by running model descriptions</td></tr>
<tr><td><a href="trace-derive">trace-derive/</a></td><td>tessera-trace-derive</td><td>#[derive(Weights)] for model descriptions</td></tr>
<tr><td><a href="turns">turns/</a></td><td>tessera-turns</td><td>Turns between models on one device, by load cost</td></tr>
</table>
