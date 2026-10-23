# Early stopping contract and focused design

Enable automatic stopping with `--auto-stop -p validation.ffm`. The existing
`-b PATIENCE` option sets a positive count of consecutive validation epochs
without a strict improvement; its default is one.

The first finite validation loss establishes the best checkpoint. A strictly
lower loss saves all model weights and optimizer state and resets patience.
Equal or higher losses count toward patience. Equal losses retain the earliest
checkpoint. The returned/saved model is the best observed validation model,
including when training reaches its iteration limit before patience expires.
Without `--auto-stop`, training returns its final model as before.

Automatic stopping requires a nonempty validation dataset, positive patience
and iteration limit, and finite monitored loss. Errors must fail explicitly
rather than disable monitoring or install an uninitialized checkpoint.

The repair stays in the existing training loop. It adds an explicit best-epoch
marker, uses strict improvement for checkpoint selection, and restores once at
the common exit. Backup storage is allocated only in automatic-stop mode. No
new public API or model format is introduced.

`make test` exercises fresh single-thread training processes, saved models,
prediction results, and the public API. Fixtures are generated in temporary
directories. Prefix comparisons include the same validation file to preserve
the existing random traversal sequence. SSE and scalar configurations are
checked separately using the existing build switches.
