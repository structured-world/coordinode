# Embedded quickstart

An application that embeds CoordiNode as a Rust library: no server, no
network. It depends only on the registry packages `coordinode-embed` and
`coordinode-core`, like any application would.

```sh
cargo run -- my-quickstart-data
```

The program:

1. opens (or creates) a database in the directory,
2. declares a unique constraint (safe to repeat),
3. runs parameterized statements,
4. commits a two-statement transaction atomically,
5. rolls another transaction back,
6. handles a duplicate as the typed `DatabaseError::UniqueViolation`,
7. closes the database by dropping it,
8. reopens the directory and checks that the committed rows are there and the
   rolled-back and refused ones are not.

It prints the run number and exits with a non-zero status if any check fails.

**The directory belongs to the program.** The default is
`coordinode-quickstart-data` in the current directory. The first run creates
the database; each later run adds its own rows under the next run number and
checks those, so running it again is safe. It never deletes anything. Give it a
directory of its own, never another application's database. Remove the
directory yourself when you are done with it.

Requirements: Rust 1.92 or newer. The guide this example goes with is
[Embedded (Rust)](../../docs/guide/embedded.md).
