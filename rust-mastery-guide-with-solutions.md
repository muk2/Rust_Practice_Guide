# Rust Mastery Guide: From Intermediate to Expert (With Solutions)

**Sources:** *Rust for Rustaceans* (Jon Gjengset) + *Rust Atomics and Locks* (Mara Bos)

---

## Part 1: The Memory Model — How Rust Thinks About Data

### 1.1 Values, Variables, and Pointers

Rust makes a precise distinction between three concepts that most languages blur together:

- **Value**: A type + an element of that type's domain. The number `6u8` *is* the mathematical integer 6 regardless of where its byte `0x06` is stored.
- **Place** (variable): A named location that can hold a value. A slot on the stack, heap, or static memory.
- **Pointer**: A value holding the address of a place. Multiple pointers can refer to the same place.

```rust
let x = 42;        // x is a place (stack slot), 42 is a value
let y = 43;        // y is a separate place
let var1 = &x;     // var1 holds a pointer value (address of x)
let mut var2 = &x; // var2 holds a copy of that same pointer value
var2 = &y;         // var2 now holds a different pointer; var1 unchanged
```

The `=` operator stores the value of the right-hand side into the place named by the left-hand side. `var1` and `var2` hold *independent copies* of pointer values.

### 1.2 Two Mental Models for Variables

**High-Level Model (Flow-Based):** Variables are names given to values as they flow through a program. Each access draws a dependency line (a "flow") from the previous access. The borrow checker verifies that all parallel flows are compatible — no two mutable flows, no mutable flow while shared flows exist.

**Low-Level Model (Slot-Based):** Variables are memory slots that may or may not contain valid values. Assignment fills the slot (dropping the old value). Access checks the slot isn't empty. `&x` points to the slot's backing memory and doesn't change when you reassign `x`.

**Use both models.** The high-level model is for reasoning about lifetimes and borrowing. The low-level model is for reasoning about unsafe code and raw pointers.

### 1.3 Memory Regions

**Stack:** Scratch space for function calls. Each call pushes a frame containing local variables and arguments. When the function returns, the frame is reclaimed. Stack frames are tied directly to lifetimes — any reference to a stack variable must have a lifetime no longer than the frame.

**Heap:** Memory not tied to any call stack. Values live until explicitly deallocated. `Box::new(value)` allocates on the heap; the `Box<T>` is a pointer to it. When the Box drops, the heap memory is freed. Heap pointers have unconstrained lifetimes. Use `Box::leak` to get `&'static` references to heap data.

**Static Memory:** Loaded from the compiled binary when the program starts. Lives for the entire execution. Includes `static` variables, string literals, and program code. The `'static` lifetime means "valid as long as static memory exists" (i.e., the whole program). `T: 'static` means T is self-sufficient — it doesn't borrow non-static values.

```rust
// 'static as a bound doesn't mean "stored in static memory"
// It means "can live for the rest of the program"
fn spawn_thread<F: FnOnce() + Send + 'static>(f: F) { /* ... */ }

// This works because the closure owns its data (no borrows):
let data = vec![1, 2, 3];
std::thread::spawn(move || println!("{:?}", data));
```

> **`const` vs `static`:** `const` has no memory location — the compiler inlines its computed value everywhere it's referenced. `static` has a fixed memory address and lives for the program's duration.

### 1.4 Exercises: Memory Foundations

**Exercise 1.1 — Predict the output:**
```rust
fn main() {
    let x = String::from("hello");
    let y = x;
    // What happens if we try to use x here?
    // println!("{}", x); // Does this compile? Why or why not?
    println!("{}", y);
}
```

<details>
<summary><b>Solution 1.1</b></summary>

The `println!("{}", x)` line does **not** compile. `String` is not `Copy`, so `let y = x` **moves** ownership from `x` to `y`. After the move, `x` is no longer valid — its "slot" is empty in the low-level model, or there are no flows from `x` in the high-level model. The compiler error is: `"value used here after move"`. Only `println!("{}", y)` works.

```rust
fn main() {
    let x = String::from("hello");
    let y = x;  // x is MOVED to y. x is now invalid.
    // println!("{}", x); // ERROR: borrow of moved value: `x`
    println!("{}", y);    // prints "hello"
}
```

If you needed both to work, you'd clone: `let y = x.clone();`
</details>

**Exercise 1.2 — Stack vs Heap:**
```rust
fn main() {
    let stack_val: i32 = 42;         // Where does this live?
    let heap_val = Box::new(42i32);  // Where does the 42 live? Where does the Box live?
    let leaked: &'static i32 = Box::leak(Box::new(99));
    // Why can we return `leaked` from this function?
}
```

<details>
<summary><b>Solution 1.2</b></summary>

- `stack_val`: The `i32` value `42` lives **on the stack** in `main`'s frame.
- `heap_val`: The `Box` pointer itself lives **on the stack**, but the `i32` value `42` it points to lives **on the heap**.
- `leaked`: `Box::new(99)` allocates `99` on the heap. `Box::leak` consumes the Box without freeing the heap memory, returning a `&'static i32`. The `99` will live on the heap for the entire program. We can return `leaked` because it has `'static` lifetime — it will never become a dangling reference, since the heap memory is intentionally never freed.

```rust
fn make_static() -> &'static i32 {
    // This works: leaked heap memory outlives any function
    Box::leak(Box::new(99))
}

fn fails() -> &i32 {
    let x = 42;
    &x  // ERROR: `x` does not live long enough (stack frame dies)
}
```
</details>

**Exercise 1.3 — Draw the flows:** Does this compile?
```rust
let mut x = Box::new(42);
let r = &x;         // flow 'a starts
if rand::random() {
    *x = 84;        // exclusive access needed
} else {
    println!("{}", r); // 'a used here
}
```

<details>
<summary><b>Solution 1.3</b></summary>

**Yes, this compiles!** The borrow checker is smart enough to realize that `r` (the shared reference) is only used in the `else` branch, while `*x = 84` (the mutable access) is only in the `if` branch. These two branches are mutually exclusive — they can never execute simultaneously. The lifetime `'a` does NOT extend into the `if` branch because `r` is never used there. There are no conflicting flows.

If you added `println!("{}", r);` **after** the if/else (at the end), it would fail to compile because `'a` would then need to extend through both branches, conflicting with the mutable access in the `if` branch.
</details>

---

## Part 2: Ownership & Borrowing — The Borrow Checker Deep Dive

### 2.1 Ownership Rules

Every value has exactly one owner. When the owner goes out of scope, the value is dropped. Moving a value transfers ownership — the old location becomes inaccessible.

**Copy types** are the exception: assigning copies bits rather than moving. A type can be `Copy` only if duplicating its bits is safe — no heap allocations, no resources to free.

```rust
let x1 = 42;               // i32 is Copy
let y1 = Box::new(84);     // Box is NOT Copy
{
    let z = (x1, y1);       // x1 copied into z, y1 MOVED into z
}                            // z dropped → drops copied x1 value AND the Box from y1
let x2 = x1;                // OK — x1 was copied, still valid
// let y2 = y1;             // ERROR — y1 was moved into z
```

### 2.2 Drop Order

Variables drop in **reverse** declaration order (later variables may reference earlier ones). Nested values (tuple fields, struct fields) drop in **source-code order** (first field first).

### 2.3 Shared References (`&T`)

- Multiple `&T` can coexist pointing to the same value
- `&T` is `Copy` — making more references is trivial
- Cannot mutate through `&T` (with exceptions: interior mutability)
- Compiler may assume the value behind `&T` won't change

### 2.4 Mutable References (`&mut T`)

- **Exclusive**: no other references (shared or mutable) to the same value can exist simultaneously
- Compiler assumes `&mut T` has exclusive access — enables powerful optimizations like noalias

**Critical rule:** If you move a value out from behind `&mut T`, you *must* leave another value in its place.

```rust
fn replace_with_84(s: &mut Box<i32>) {
    // let was = *s;                    // ERROR: can't move out
    let was = std::mem::take(s);        // OK: replaces with default
    *s = was;                           // OK: put it back
    let mut r = Box::new(84);
    std::mem::swap(s, &mut r);          // OK: swap with another owned value
}
```

### 2.5 Interior Mutability

Some types allow mutation through `&T` by providing safety through runtime checks or CPU-level guarantees:

| Type | Thread-Safe? | Mechanism | Use Case |
|------|-------------|-----------|----------|
| `Cell<T>` | No | Copy in/out (no references given out) | Single-thread, simple values |
| `RefCell<T>` | No | Runtime borrow tracking (panics on violation) | Single-thread, complex values |
| `Mutex<T>` | Yes | OS-level locking (blocks on contention) | Multi-thread shared state |
| `RwLock<T>` | Yes | Multiple readers OR one writer | Multi-thread, read-heavy |
| `AtomicU32`, etc. | Yes | CPU atomic instructions | Multi-thread, simple values |
| `UnsafeCell<T>` | — | Raw primitive (unsafe) | Building other interior-mutable types |

### 2.6 Exercises: Ownership & Borrowing

**Exercise 2.1 — Fix the borrow checker error:**
```rust
fn main() {
    let mut data = vec![1, 2, 3];
    let first = &data[0];   // shared borrow
    data.push(4);            // mutable borrow — ERROR
    println!("{}", first);   // shared borrow used here
}
```

<details>
<summary><b>Solution 2.1</b></summary>

The problem: `&data[0]` borrows `data` immutably, `push` borrows it mutably, and then we use `first` again. The mutable and immutable borrows overlap. Three approaches:

```rust
// Approach 1: Finish using the reference before mutating
fn main() {
    let mut data = vec![1, 2, 3];
    let first = data[0];   // COPY the i32 (no borrow held)
    data.push(4);
    println!("{}", first);  // prints 1
}

// Approach 2: Clone first, then mutate
fn main() {
    let mut data = vec![1, 2, 3];
    let first = data[0].clone();
    data.push(4);
    println!("{}", first);
}

// Approach 3: Reorder to end the shared borrow before mutating
fn main() {
    let mut data = vec![1, 2, 3];
    let first = &data[0];
    println!("{}", first);  // use the reference FIRST
    data.push(4);           // now mutate (no active shared borrows)
}
```

The real reason `push` is dangerous here: `push` might reallocate the Vec's buffer, invalidating all existing references into it. Rust's borrow checker prevents this class of use-after-free bugs.
</details>

**Exercise 2.2 — Interior mutability counter:**
```rust
use std::cell::Cell;

struct Counter {
    // What goes here?
}

impl Counter {
    fn new() -> Self { /* ... */ }
    fn increment(&self) { /* note: &self, not &mut self */ }
    fn get(&self) -> u32 { /* ... */ }
}
```

<details>
<summary><b>Solution 2.2</b></summary>

```rust
use std::cell::Cell;

struct Counter {
    count: Cell<u32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }

    fn increment(&self) {
        // Cell::set replaces the value. Cell::get copies it out.
        // No references to the inner value are ever created.
        self.count.set(self.count.get() + 1);
    }

    fn get(&self) -> u32 {
        self.count.get()
    }
}

fn main() {
    let c = Counter::new();
    c.increment();
    c.increment();
    c.increment();
    assert_eq!(c.get(), 3);
}
```

`Cell<u32>` works because `u32` is `Copy`. The cell never hands out a reference to the inner value — only copies it in and out. This makes mutation through `&self` safe because there can never be a dangling or aliased reference to the inner data.
</details>

**Exercise 2.3 — Implement `std::mem::replace`:**
```rust
fn my_replace<T>(dest: &mut T, src: T) -> T {
    todo!()
}
```

<details>
<summary><b>Solution 2.3</b></summary>

```rust
fn my_replace<T>(dest: &mut T, mut src: T) -> T {
    std::mem::swap(dest, &mut src);
    src  // src now holds what was in dest
}

// How std::mem::swap works conceptually (actual impl uses unsafe):
// It exchanges the values behind two &mut T references.
// After swap: dest has new value, src has old value.
// We return src (the old value).

// Why this is safe:
// - dest is &mut T: we have exclusive access, guaranteed one mutable ref
// - src is T: we own it outright
// - After swap, dest still holds a valid T (the new value)
// - src holds a valid T (the old value) which we return
// - No moment exists where any T is uninitialized or double-freed
```
</details>

**Exercise 2.4 — Reason about lifetimes with holes:**
```rust
let mut x = Box::new(42);
let mut z = &x;
for i in 0..100 {
    println!("{}", z);  // use 'a
    x = Box::new(i);    // move x — does this kill 'a?
    z = &x;             // restart 'a
}
println!("{}", z);
```

<details>
<summary><b>Solution 2.4</b></summary>

**Yes, this compiles.** Here's why:

1. `z = &x` — lifetime `'a` starts, `z` references `x`
2. `println!("{}", z)` — `'a` is used here (last use before invalidation)
3. `x = Box::new(i)` — `x` is reassigned. The OLD `'a` ends because the old `Box` is dropped. The borrow checker allows this because `'a`'s last use was `println` above.
4. `z = &x` — A NEW lifetime starts. `z` now references the new `x`.
5. Loop repeats: step 2 uses the new `'a`, step 3 kills it, step 4 restarts it.
6. Final `println!("{}", z)` — uses the `'a` from the last iteration.

The lifetime has **holes** — it's intermittently invalid between uses. Each loop iteration creates a fresh borrow. The old Box (42) is dropped on the first `x = Box::new(i)`, and that's fine because we finished using the reference to it.
</details>

---

## Part 3: Lifetimes — The Full Picture

### 3.1 What Lifetimes Really Are

A lifetime is a **name for a region of code** that a reference must be valid for. It's NOT the same as a scope — lifetimes can have holes, can be non-contiguous, and can fork at branches.

### 3.2 Lifetime Variance

- **Covariant** (`&'a T`): Can substitute `'long` where `'short` is expected
- **Invariant** (`&'a mut T` in T): Cannot substitute — must be exactly the same lifetime
- **Contravariant** (`fn(&'a T)`): Reversed — can substitute `'short` where `'long` is expected

> **Rule of thumb:** Use separate lifetime parameters when different references may have different lifetimes. Excess invariance from combining lifetimes is a common source of confusing borrow checker errors.

### 3.3 Exercises: Lifetimes

**Exercise 3.1 — Annotate lifetimes:**
```rust
struct Important {
    content: &str,
}

fn longest(x: &str, y: &str) -> &str {
    if x.len() > y.len() { x } else { y }
}
```

<details>
<summary><b>Solution 3.1</b></summary>

```rust
struct Important<'a> {
    content: &'a str,
}
// The struct borrows a string slice, so we need a lifetime parameter.
// 'a says: "this struct can't outlive the str it references."

fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
// Both inputs share lifetime 'a, and the return value also has 'a.
// The compiler infers 'a as the SHORTER of the two input lifetimes.
// This means the returned reference is valid for at least as long as
// both inputs are valid.

// Usage:
fn main() {
    let s1 = String::from("long string");
    let result;
    {
        let s2 = String::from("hi");
        result = longest(s1.as_str(), s2.as_str());
        println!("{}", result); // OK: both s1 and s2 are alive
    }
    // println!("{}", result); // ERROR: s2 was dropped, 'a has ended
}
```
</details>

**Exercise 3.2 — Why doesn't this compile? Fix it.**
```rust
fn main() {
    let mut s = String::from("hello");
    let r1 = &s;
    let r2 = &s;
    let r3 = &mut s;
    println!("{}, {}, {}", r1, r2, r3);
}
```

<details>
<summary><b>Solution 3.2</b></summary>

Can't have `&s` (shared) and `&mut s` (exclusive) alive simultaneously. Fix: end the shared borrows before taking the mutable one.

```rust
fn main() {
    let mut s = String::from("hello");
    let r1 = &s;
    let r2 = &s;
    println!("{}, {}", r1, r2);  // last use of r1, r2 — their lifetimes end

    let r3 = &mut s;             // now OK: no active shared borrows
    println!("{}", r3);
}
```
</details>

**Exercise 3.3 — Implement an iterator that borrows from a struct:**
```rust
struct StrSplit<'a> {
    remainder: Option<&'a str>,
    delimiter: char,
}

impl<'a> Iterator for StrSplit<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
```

<details>
<summary><b>Solution 3.3</b></summary>

```rust
struct StrSplit<'a> {
    remainder: Option<&'a str>,
    delimiter: char,
}

impl<'a> StrSplit<'a> {
    fn new(haystack: &'a str, delimiter: char) -> Self {
        StrSplit {
            remainder: Some(haystack),
            delimiter,
        }
    }
}

impl<'a> Iterator for StrSplit<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        // Take the remainder out (leaves None in its place)
        let remainder = self.remainder.as_mut()?;
        if let Some(pos) = remainder.find(self.delimiter) {
            let before = &remainder[..pos];
            *remainder = &remainder[pos + self.delimiter.len_utf8()..];
            Some(before)
        } else {
            // No more delimiters — return everything that's left
            self.remainder.take()
        }
    }
}

fn main() {
    let text = "hello world foo bar";
    let splits: Vec<&str> = StrSplit::new(text, ' ').collect();
    assert_eq!(splits, vec!["hello", "world", "foo", "bar"]);
}
```

Key insight: The iterator yields `&'a str` — references that borrow from the *original* `haystack`, not from the iterator itself. The `'a` on the struct ties the output lifetime to the input string's lifetime.
</details>

---

## Part 4: Types, Traits, and Dispatch

### 4.1 Type Layout in Memory

- **repr(C)**: Fields in declaration order with padding for alignment. Predictable, compatible with C.
- **repr(Rust)** (default): Compiler may reorder fields to minimize padding.
- **repr(packed)**: No padding. Smaller but potentially slower.
- **repr(transparent)**: Single-field struct same layout as inner field.

### 4.2 Static Dispatch (Monomorphization)

`fn foo<T: Trait>(x: T)` → compiler generates a separate copy for every concrete `T`. Zero overhead, enables inlining.

### 4.3 Dynamic Dispatch (Trait Objects)

`&dyn Trait` → wide pointer: (data pointer, vtable pointer). Single copy of code, can't inline across vtable.

**Object Safety:** A trait can be `dyn Trait` only if: no methods return `Self`, no generic methods, no static methods.

### 4.4 Send and Sync

- **`Send`**: Safe to transfer to another thread
- **`Sync`**: Safe to share references across threads (`T` is Sync iff `&T` is Send)

### 4.5 Exercises: Types & Traits

**Exercise 4.1 — Static to dynamic dispatch:**
```rust
fn print_all<I: Iterator<Item = i32>>(iter: I) {
    for item in iter {
        println!("{}", item);
    }
}
```

<details>
<summary><b>Solution 4.1</b></summary>

```rust
// Dynamic dispatch version — accepts any iterator as a trait object
fn print_all(iter: &mut dyn Iterator<Item = i32>) {
    for item in iter {
        println!("{}", item);
    }
}

fn main() {
    let mut v = vec![1, 2, 3].into_iter();
    print_all(&mut v);

    let mut r = (0..5);
    print_all(&mut r);
}
```

Note: We must use `&mut dyn Iterator` (a reference to the trait object) because `dyn Iterator` is unsized. We need `&mut` (not `&`) because `Iterator::next` takes `&mut self`. Could also use `Box<dyn Iterator<Item = i32>>`.
</details>

**Exercise 4.2 — Object safety:** Which traits can be `dyn Trait`?
```rust
trait A { fn foo(&self); }
trait B { fn foo(&self) -> Self; }
trait C { fn foo<T>(&self, x: T); }
trait D { fn foo(&self); fn bar() where Self: Sized; }
```

<details>
<summary><b>Solution 4.2</b></summary>

- **A**: ✅ Object-safe. `foo` takes `&self`, no generics, no `Self` in return.
- **B**: ❌ Not object-safe. `foo` returns `Self`. If called through a vtable, the compiler doesn't know the concrete type to return.
- **C**: ❌ Not object-safe. `foo` is generic over `T`. The vtable would need infinite entries (one per `T`).
- **D**: ✅ Object-safe! `bar()` has `where Self: Sized`, which exempts it from the object-safety check. Only `foo` needs to be object-safe, and it is. When using `dyn D`, `bar()` simply isn't available.
</details>

**Exercise 4.3 — Plugin system:**

<details>
<summary><b>Solution 4.3</b></summary>

```rust
trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, input: &str) -> String;
}

struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginManager {
    fn new() -> Self {
        PluginManager { plugins: Vec::new() }
    }
    fn register(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }
    fn run_all(&self, input: &str) -> Vec<String> {
        self.plugins.iter().map(|p| p.execute(input)).collect()
    }
}

// Example plugins
struct UpperPlugin;
impl Plugin for UpperPlugin {
    fn name(&self) -> &str { "upper" }
    fn execute(&self, input: &str) -> String { input.to_uppercase() }
}

struct ReversePlugin;
impl Plugin for ReversePlugin {
    fn name(&self) -> &str { "reverse" }
    fn execute(&self, input: &str) -> String { input.chars().rev().collect() }
}

fn main() {
    let mut mgr = PluginManager::new();
    mgr.register(Box::new(UpperPlugin));
    mgr.register(Box::new(ReversePlugin));
    let results = mgr.run_all("hello");
    assert_eq!(results, vec!["HELLO".to_string(), "olleh".to_string()]);
}
```
</details>

**Exercise 4.4 — Memory layout calculation:**
```rust
struct Mixed { a: u8, b: u64, c: u16, d: u8, e: u32 }
```

<details>
<summary><b>Solution 4.4</b></summary>

**With `repr(C)` (fields in declaration order):**
```
Offset  Field  Size  Padding after
0       a      1     7 (align b to 8)
8       b      8     0
16      c      2     2 (align d... actually d is u8, no padding needed)
                     but wait — d is u8 at offset 18, then e is u32 needing 4-byte alignment
18      d      1     1 (align e to offset 20, which is 4-byte aligned)
20      e      4     0
Total: 24 bytes. Alignment: 8 (from u64). 24 is a multiple of 8 ✓
```

**With default `repr(Rust)` (compiler reorders):**
Optimal ordering by decreasing alignment: b(u64), e(u32), c(u16), a(u8), d(u8)
```
Offset  Field  Size
0       b      8
8       e      4
12      c      2
14      a      1
15      d      1
Total: 16 bytes. Zero padding!
```
The Rust compiler saves **8 bytes** (33%) by reordering fields.
</details>

---

## Part 5: Concurrency Foundations

### 5.1 Threads in Rust

```rust
use std::thread;

// Scoped threads — can borrow local variables
let data = vec![1, 2, 3];
thread::scope(|s| {
    s.spawn(|| println!("length: {}", data.len()));
    s.spawn(|| { for n in &data { println!("{n}"); } });
}); // all scoped threads auto-joined here
```

### 5.2 Mutex and Condvar

`Mutex<T>` dynamically enforces exclusive access. `Condvar` allows threads to sleep waiting for a condition.

### 5.3 Exercises: Concurrency Basics

**Exercise 5.1 — Thread-safe counter:**

<details>
<summary><b>Solution 5.1</b></summary>

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0u64));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let mut num = counter.lock().unwrap();
                *num += 1;
                // MutexGuard dropped here → lock released
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(*counter.lock().unwrap(), 10_000);
    println!("Final count: {}", *counter.lock().unwrap());
}
```

Key points: `Arc` provides shared ownership across threads. `Mutex` provides exclusive mutable access. The `MutexGuard` returned by `lock()` auto-unlocks via `Drop`.
</details>

**Exercise 5.2 — Producer-Consumer with Condvar:**

<details>
<summary><b>Solution 5.2</b></summary>

```rust
use std::collections::VecDeque;
use std::sync::{Mutex, Condvar, Arc};
use std::thread;

struct BoundedQueue<T> {
    inner: Mutex<VecDeque<T>>,
    capacity: usize,
    not_empty: Condvar,
    not_full: Condvar,
}

impl<T> BoundedQueue<T> {
    fn new(capacity: usize) -> Self {
        BoundedQueue {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    fn push(&self, item: T) {
        let mut queue = self.inner.lock().unwrap();
        while queue.len() >= self.capacity {
            queue = self.not_full.wait(queue).unwrap();
        }
        queue.push_back(item);
        self.not_empty.notify_one();
    }

    fn pop(&self) -> T {
        let mut queue = self.inner.lock().unwrap();
        while queue.is_empty() {
            queue = self.not_empty.wait(queue).unwrap();
        }
        let item = queue.pop_front().unwrap();
        self.not_full.notify_one();
        item
    }
}

fn main() {
    let queue = Arc::new(BoundedQueue::new(5));

    // Producers
    let q = Arc::clone(&queue);
    let producer = thread::spawn(move || {
        for i in 0..20 {
            q.push(i);
            println!("Produced: {i}");
        }
    });

    // Consumer
    let q = Arc::clone(&queue);
    let consumer = thread::spawn(move || {
        for _ in 0..20 {
            let item = q.pop();
            println!("Consumed: {item}");
        }
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}
```

The `while` loops (not `if`) handle spurious wakeups. `Condvar::wait` atomically releases the mutex and sleeps, then re-locks before returning.
</details>

---

## Part 6: Atomics — The Foundation of Lock-Free Programming

### 6.1 Overview

Atomic operations are **indivisible**. Available types: `AtomicBool`, `AtomicI32`, `AtomicUsize`, `AtomicPtr<T>`, etc. All take an `Ordering` argument.

### 6.2 Operations

- **Load/Store:** Basic read/write
- **Fetch-and-modify:** `fetch_add`, `fetch_sub`, `fetch_or`, `swap` — modify and return OLD value
- **Compare-and-exchange (CAS):** Check if value equals expected, and if so, replace with new

### 6.3 Exercises: Atomics

**Exercise 6.1 — Atomic statistics counter:**

<details>
<summary><b>Solution 6.1</b></summary>

```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

struct Stats {
    total: AtomicU64,
    count: AtomicUsize,
}

impl Stats {
    fn new() -> Self {
        Stats {
            total: AtomicU64::new(0),
            count: AtomicUsize::new(0),
        }
    }

    fn add_sample(&self, value: u64) {
        self.total.fetch_add(value, Relaxed);
        self.count.fetch_add(1, Relaxed);
    }

    fn snapshot(&self) -> (u64, usize) {
        // IMPORTANT: total and count may not be perfectly in sync!
        // One thread could have added to total but not yet incremented count.
        // For perfect consistency, you'd need a Mutex or a single atomic
        // holding both values (e.g., pack into a u128 or use a lock).
        let total = self.total.load(Relaxed);
        let count = self.count.load(Relaxed);
        (total, count)
    }
}

fn main() {
    use std::thread;
    let stats = Stats::new();
    thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for i in 0..100 {
                    stats.add_sample(i);
                }
            });
        }
    });
    let (total, count) = stats.snapshot();
    println!("total={total}, count={count}, avg={}", total / count as u64);
}
```

`Relaxed` is fine here because we only care about each individual variable's correctness, not the ordering between `total` and `count`.
</details>

**Exercise 6.2 — Lock-free lazy initialization:**

<details>
<summary><b>Solution 6.2</b></summary>

```rust
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

fn get_cached_value() -> u64 {
    static CACHE: AtomicU64 = AtomicU64::new(0);

    let mut val = CACHE.load(Relaxed);
    if val == 0 {
        val = expensive_computation();
        CACHE.store(val, Relaxed);
    }
    val
}

fn expensive_computation() -> u64 {
    println!("Computing...");
    42
}

fn main() {
    // First call computes
    println!("{}", get_cached_value()); // prints "Computing..." then "42"
    // Second call uses cache
    println!("{}", get_cached_value()); // prints "42" (no "Computing...")
}
```

**Note:** Multiple threads might race to compute simultaneously and overwrite each other. This is fine if the computation is deterministic (both get the same result). If the computation is expensive and you want only one thread to do it, use `std::sync::OnceLock` instead.
</details>

---

## Part 7: Memory Ordering — The Hardest Part

### 7.1 The Hierarchy

| Ordering | Guarantees |
|----------|-----------|
| `Relaxed` | Atomicity only. No cross-variable ordering. Per-variable total modification order. |
| `Release` (stores) + `Acquire` (loads) | Creates happens-before between release-store and acquire-load of same variable |
| `SeqCst` | All SeqCst ops have a single global total order all threads agree on |

### 7.2 Release/Acquire Pattern

```rust
// Thread 1 (producer):
DATA.store(123, Relaxed);
READY.store(true, Release);  // "I release everything I did above"

// Thread 2 (consumer):
while !READY.load(Acquire) {} // "I acquire what the releaser did"
println!("{}", DATA.load(Relaxed)); // GUARANTEED to see 123
```

### 7.3 Exercises: Memory Ordering

**Exercise 7.1 — Identify the bug:**
```rust
static DATA: AtomicU64 = AtomicU64::new(0);
static READY: AtomicBool = AtomicBool::new(false);

// Thread 1:
DATA.store(42, Relaxed);
READY.store(true, Relaxed);  // BUG!

// Thread 2:
if READY.load(Relaxed) {     // BUG!
    println!("{}", DATA.load(Relaxed)); // Could print 0!
}
```

<details>
<summary><b>Solution 7.1</b></summary>

**Bug:** Both READY operations use `Relaxed`. With relaxed ordering, there's no happens-before relationship between the two threads. Thread 2 might see `READY = true` but still load the old `DATA = 0` because relaxed operations on different variables have no ordering guarantees. The processor/compiler could reorder the stores or loads.

**Fix:** Use Release on the store, Acquire on the load:
```rust
// Thread 1:
DATA.store(42, Relaxed);
READY.store(true, Release);  // Release: everything before this is visible to acquirer

// Thread 2:
if READY.load(Acquire) {     // Acquire: see everything the releaser did
    println!("{}", DATA.load(Relaxed)); // Now GUARANTEED to print 42
}
```

The Release-Acquire pair on READY creates a happens-before relationship: Thread 1's `DATA.store(42)` happens-before Thread 2's `DATA.load()`.
</details>

**Exercise 7.2 — Build a one-shot channel:**

<details>
<summary><b>Solution 7.2</b></summary>

```rust
use std::sync::atomic::{AtomicBool, Ordering::{Acquire, Release}};
use std::cell::UnsafeCell;

pub struct OneShotChannel<T> {
    data: UnsafeCell<Option<T>>,
    ready: AtomicBool,
}

// SAFETY: We guarantee that data is only written before ready=true (by sender)
// and only read after ready=true (by receiver). The Release/Acquire on `ready`
// creates a happens-before relationship ensuring the receiver sees the data.
unsafe impl<T: Send> Sync for OneShotChannel<T> {}

impl<T> OneShotChannel<T> {
    pub fn new() -> Self {
        OneShotChannel {
            data: UnsafeCell::new(None),
            ready: AtomicBool::new(false),
        }
    }

    /// Send a value. Must be called at most once.
    /// Calling more than once is a logic error (second call is silently ignored).
    pub fn send(&self, value: T) {
        // SAFETY: No other thread can access data because ready is still false.
        // Only one sender should call this (not enforced here — see book for typed version).
        unsafe { *self.data.get() = Some(value) };
        self.ready.store(true, Release);
        // Release ordering ensures the data write above is visible
        // to any thread that does an Acquire load of `ready`.
    }

    /// Try to receive. Returns Some(T) if data is ready.
    pub fn try_recv(&self) -> Option<T> {
        if self.ready.load(Acquire) {
            // Acquire ensures we see the data the sender wrote.
            // SAFETY: ready is true, so data has been written.
            // We take() it so subsequent calls return None.
            unsafe { (*self.data.get()).take() }
        } else {
            None
        }
    }
}

fn main() {
    let channel = OneShotChannel::new();

    std::thread::scope(|s| {
        s.spawn(|| {
            channel.send(42);
        });
        s.spawn(|| {
            // Spin until we get the value
            loop {
                if let Some(val) = channel.try_recv() {
                    assert_eq!(val, 42);
                    println!("Received: {val}");
                    break;
                }
                std::hint::spin_loop();
            }
        });
    });
}
```

**Why this is safe:**
1. `send` writes data, then Release-stores `true` to `ready`
2. `try_recv` Acquire-loads `ready`. If true, the happens-before relationship guarantees the data write is visible.
3. `take()` ensures the value can only be received once.
4. `T: Send` bound ensures the type is safe to transfer between threads.
</details>

---

## Part 8: Building Synchronization Primitives

### 8.1 Spin Lock

```rust
use std::sync::atomic::{AtomicBool, Ordering::{Acquire, Release}};
use std::cell::UnsafeCell;

pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub const fn new(value: T) -> Self {
        SpinLock { locked: AtomicBool::new(false), data: UnsafeCell::new(value) }
    }

    pub fn lock(&self) -> SpinGuard<T> {
        while self.locked.swap(true, Acquire) {
            std::hint::spin_loop();
        }
        SpinGuard { lock: self }
    }
}

pub struct SpinGuard<'a, T> { lock: &'a SpinLock<T> }

impl<T> std::ops::Deref for SpinGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { &*self.lock.data.get() } }
}
impl<T> std::ops::DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.lock.data.get() } }
}
impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) { self.lock.locked.store(false, Release); }
}
```

### 8.2 Exercises: Build Your Own Primitives

**Exercise 8.1 — Add `try_lock`:**

<details>
<summary><b>Solution 8.1</b></summary>

```rust
impl<T> SpinLock<T> {
    pub fn try_lock(&self) -> Option<SpinGuard<T>> {
        // Attempt ONE swap. If it returns false, we got the lock.
        // If it returns true, lock was already held — return None.
        if self.locked.swap(true, Acquire) {
            None  // Already locked
        } else {
            Some(SpinGuard { lock: self })  // We got it
        }
    }
}

// Alternative using compare_exchange:
impl<T> SpinLock<T> {
    pub fn try_lock_v2(&self) -> Option<SpinGuard<T>> {
        self.locked
            .compare_exchange(false, true, Acquire, Relaxed)
            .ok()
            .map(|_| SpinGuard { lock: self })
    }
}
```
</details>

**Exercise 8.2 — Thread-safe lazy initializer:**

<details>
<summary><b>Solution 8.2</b></summary>

```rust
use std::sync::atomic::{AtomicU8, Ordering::{Acquire, Release, Relaxed}};
use std::cell::UnsafeCell;

const UNINIT: u8 = 0;
const INITIALIZING: u8 = 1;
const READY: u8 = 2;

pub struct Lazy<T> {
    state: AtomicU8,
    data: UnsafeCell<Option<T>>,
}

unsafe impl<T: Send + Sync> Sync for Lazy<T> {}

impl<T> Lazy<T> {
    pub fn new() -> Self {
        Lazy {
            state: AtomicU8::new(UNINIT),
            data: UnsafeCell::new(None),
        }
    }

    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
        // Fast path: already initialized
        if self.state.load(Acquire) == READY {
            return unsafe { (*self.data.get()).as_ref().unwrap() };
        }

        // Try to become the initializer
        match self.state.compare_exchange(UNINIT, INITIALIZING, Acquire, Acquire) {
            Ok(_) => {
                // We won the race — initialize
                unsafe { *self.data.get() = Some(f()) };
                self.state.store(READY, Release);
                unsafe { (*self.data.get()).as_ref().unwrap() }
            }
            Err(_) => {
                // Someone else is initializing or already done — spin
                while self.state.load(Acquire) != READY {
                    std::hint::spin_loop();
                }
                unsafe { (*self.data.get()).as_ref().unwrap() }
            }
        }
    }
}

fn main() {
    let lazy = Lazy::new();
    std::thread::scope(|s| {
        for i in 0..10 {
            s.spawn(|| {
                let val = lazy.get_or_init(|| {
                    println!("Initializing from thread {i}!");
                    42
                });
                assert_eq!(*val, 42);
            });
        }
    });
}
// Only one thread prints "Initializing..." — the rest spin-wait.
```

In production, prefer `std::sync::OnceLock` which does this correctly with proper blocking.
</details>

**Exercise 8.3 — Simple MPSC channel:**

<details>
<summary><b>Solution 8.3</b></summary>

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar};

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        queue: Mutex::new(VecDeque::new()),
        available: Condvar::new(),
    });
    (
        Sender { shared: Arc::clone(&shared) },
        Receiver { shared },
    )
}

struct Shared<T> {
    queue: Mutex<VecDeque<T>>,
    available: Condvar,
}

pub struct Sender<T> {
    shared: Arc<Shared<T>>,
}

pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) {
        self.shared.queue.lock().unwrap().push_back(value);
        self.shared.available.notify_one();
    }
}

// Sender is Clone — multiple producers can send
impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Sender { shared: Arc::clone(&self.shared) }
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> T {
        let mut queue = self.shared.queue.lock().unwrap();
        loop {
            if let Some(value) = queue.pop_front() {
                return value;
            }
            queue = self.shared.available.wait(queue).unwrap();
        }
    }
}

fn main() {
    let (tx, rx) = channel();
    let tx2 = tx.clone();

    std::thread::spawn(move || {
        for i in 0..5 { tx.send(format!("from tx1: {i}")); }
    });
    std::thread::spawn(move || {
        for i in 0..5 { tx2.send(format!("from tx2: {i}")); }
    });

    for _ in 0..10 {
        println!("Received: {}", rx.recv());
    }
}
```

This mirrors the design from *Rust Atomics and Locks* Chapter 5. The `Condvar` + `Mutex<VecDeque>` approach is simple, correct, and suitable for many real-world cases. For higher performance, look at lock-free channel designs in crates like `crossbeam-channel`.
</details>

---

## Part 9: Capstone Projects

### Project 1: Thread Pool

<details>
<summary><b>Full Solution</b></summary>

```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

struct Worker {
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        let workers: Vec<Worker> = (0..size)
            .map(|id| {
                let receiver = Arc::clone(&receiver);
                let handle = thread::spawn(move || loop {
                    // Lock the receiver, get a job, then immediately unlock
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => {
                            println!("Worker {id} executing job");
                            job();
                        }
                        Err(_) => {
                            // Channel closed — shut down
                            println!("Worker {id} shutting down");
                            break;
                        }
                    }
                });
                Worker { id, handle: Some(handle) }
            })
            .collect();

        ThreadPool { workers, sender: Some(sender) }
    }

    pub fn execute<F: FnOnce() + Send + 'static>(&self, f: F) {
        self.sender.as_ref().unwrap().send(Box::new(f)).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Drop sender → closes channel → workers see Err and exit
        drop(self.sender.take());
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                handle.join().unwrap();
            }
        }
    }
}

fn main() {
    let pool = ThreadPool::new(4);
    for i in 0..8 {
        pool.execute(move || {
            println!("Task {i} running on {:?}", thread::current().id());
            thread::sleep(std::time::Duration::from_millis(100));
        });
    }
    // Pool is dropped here → waits for all tasks to finish
}
```
</details>

### Project 2: Lock-Free Stack

<details>
<summary><b>Full Solution</b></summary>

```rust
use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
}

// SAFETY: The stack's operations use atomic CAS to maintain consistency.
// T: Send is required because values move between threads.
unsafe impl<T: Send> Send for LockFreeStack<T> {}
unsafe impl<T: Send> Sync for LockFreeStack<T> {}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        LockFreeStack { head: AtomicPtr::new(ptr::null_mut()) }
    }

    pub fn push(&self, val: T) {
        let new_node = Box::into_raw(Box::new(Node {
            data: val,
            next: ptr::null_mut(),
        }));
        loop {
            let old_head = self.head.load(Ordering::Relaxed);
            unsafe { (*new_node).next = old_head; }
            // CAS: if head still equals old_head, replace with new_node
            if self.head
                .compare_exchange_weak(old_head, new_node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            // If CAS failed, another thread modified head — retry
        }
    }

    pub fn pop(&self) -> Option<T> {
        loop {
            let old_head = self.head.load(Ordering::Acquire);
            if old_head.is_null() {
                return None;
            }
            let next = unsafe { (*old_head).next };
            if self.head
                .compare_exchange_weak(old_head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // SAFETY: We successfully removed old_head from the stack.
                // No other thread will pop this same node.
                let node = unsafe { Box::from_raw(old_head) };
                return Some(node.data);
            }
            // CAS failed — another thread popped first — retry
        }
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

fn main() {
    let stack = LockFreeStack::new();
    std::thread::scope(|s| {
        // 4 threads push
        for i in 0..4 {
            s.spawn(|| { for j in 0..100 { stack.push(i * 100 + j); } });
        }
    });
    let mut count = 0;
    while stack.pop().is_some() { count += 1; }
    assert_eq!(count, 400);
    println!("All {count} items popped successfully.");
}
```

> **Warning:** This has the ABA problem. For production use, consider epoch-based reclamation (`crossbeam-epoch`) or hazard pointers.
</details>

---

## Appendix A: Common Borrow Checker Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| "cannot borrow as mutable because also borrowed as immutable" | Overlapping `&T` and `&mut T` | End shared borrow before mutable borrow |
| "does not live long enough" | Reference outlives data | Use owned types, `Arc`, or adjust lifetimes |
| "cannot move out of borrowed content" | Taking ownership through a reference | `.clone()`, `mem::take()`, or `mem::replace()` |
| "value used after move" | Using a value after transferring ownership | Clone before moving, or restructure |
| "`T` doesn't implement `Send`" | Sending non-Send type across threads | `Arc` instead of `Rc`, atomics instead of `Cell` |

---

## Part 10: Modern Rust Tooling & Environment (2024–2025)

### 10.1 The Rust 2024 Edition (Stabilized in Rust 1.85, Feb 2025)

The 2024 edition is the largest edition release yet. Key changes:

**Let chains** (stable in 1.88): Chain `let` patterns with `&&` inside `if` and `while`:
```rust
// edition = "2024"
if let Some(x) = opt && let Ok(y) = x.parse::<i32>() && y > 0 {
    println!("positive: {y}");
}
```

**`unsafe` extern blocks**: `extern` blocks now require the `unsafe` keyword, making FFI boundaries explicit.

**`unsafe_op_in_unsafe_fn` warning by default**: Unsafe functions now require explicit `unsafe {}` blocks inside them, preventing accidental unsafe operations.

**RPIT lifetime capture changes**: `impl Trait` in return position now captures all in-scope lifetimes by default unless you use `use<..>` to specify.

**Tail expression temporary scope changes**: Temporaries in tail expressions now drop more predictably.

To migrate: `cargo fix --edition` then set `edition = "2024"` in `Cargo.toml`.

### 10.2 Recent Rust Releases (Key Features)

**Rust 1.84 (Jan 2025):**
- MSRV-aware resolver (Cargo resolver v3) — automatically selects dependency versions compatible with your minimum supported Rust version
- Strict provenance APIs for pointer-integer casts (`with_addr`, `map_addr`) — better for Miri and CHERI

**Rust 1.85 (Feb 2025):**
- 2024 Edition stabilized
- Rustdoc combined doctests (significantly faster doc testing)
- Rustfmt style editions (format independently from Rust edition)

**Rust 1.88 (Jun 2025):**
- `let_chains` stabilized (2024 edition only)
- Naked functions (`#[unsafe(naked)]`)
- `cfg(true)` / `cfg(false)` boolean literals
- Cargo automatic cache garbage collection (cleans `~/.cargo` of unused dependencies: 3 months for registry, 1 month for local)

### 10.3 Essential Development Tools

**rust-analyzer** — The standard Rust LSP. Provides real-time error checking, completion, go-to-definition, refactoring, and inlay hints in any LSP-capable editor (VS Code, Neovim, Emacs, Helix).
```bash
rustup component add rust-analyzer
```

**Clippy** — The official linter. Catches hundreds of common mistakes and anti-patterns:
```bash
cargo clippy -- -W clippy::all -W clippy::pedantic
# In CI, fail on warnings:
cargo clippy -- -D warnings
```

**Miri** — The MIR interpreter. Detects undefined behavior in unsafe code: use-after-free, data races, alignment violations, Stacked/Tree Borrows violations, memory leaks.
```bash
rustup +nightly component add miri
cargo +nightly miri test
# With nextest for parallelism:
cargo +nightly miri nextest run -j4
```
Miri is **essential** when writing any unsafe code. It catches bugs that tests alone will never find. Supports experimental Tree Borrows aliasing model (`-Zmiri-tree-borrows`).

**cargo-nextest** — Next-generation test runner. Each test runs in its own process, enabling better failure isolation, flaky test detection, and parallel Miri execution:
```bash
cargo install cargo-nextest
cargo nextest run
```

**cargo-audit** — Checks dependencies for known security vulnerabilities:
```bash
cargo install cargo-audit
cargo audit
```

**cargo-deny** — Enforces policies on dependencies: licenses, banned crates, duplicate versions, vulnerability advisories:
```bash
cargo install cargo-deny
cargo deny check
```

**cargo-machete** — Detects unused dependencies in your `Cargo.toml`:
```bash
cargo install cargo-machete
cargo machete
```

**cargo-wizard** — Optimizes your project configuration for build times, runtime performance, or binary size:
```bash
cargo install cargo-wizard
cargo wizard
```

**cargo-semver-checks** — Lints your API for semver violations before publishing (being integrated into Cargo itself):
```bash
cargo install cargo-semver-checks
cargo semver-checks check-release
```

### 10.4 Profiling & Performance

**cargo-flamegraph** — Generate flamegraphs from your Rust binaries:
```bash
cargo install flamegraph
cargo flamegraph --bin my_binary
```

**cargo-bench** + **criterion** — Statistically rigorous benchmarking:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

**DHAT** (via `dhat` crate) — Heap profiling that works with `#[global_allocator]`.

### 10.5 Recommended Editor Setup (2025)

For your Emacs + eglot setup, ensure `rust-analyzer` is configured:

```elisp
;; In your Emacs config
(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs
    '(rust-mode . ("rust-analyzer"))))

;; Enable clippy on save
(setq-default eglot-workspace-configuration
  '(:rust-analyzer
    (:checkOnSave (:command "clippy")
     :cargo (:features "all"))))
```

For **VS Code**: Install the `rust-analyzer` extension. Configure in `settings.json`:
```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.cargo.features": "all"
}
```

### 10.6 CI Pipeline Template

```yaml
# .github/workflows/rust.yml
name: Rust CI
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test
      - run: cargo install cargo-nextest && cargo nextest run
      - run: cargo install cargo-audit && cargo audit

  miri:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri
      - run: cargo +nightly miri test
```

### 10.7 Key Crates for Concurrency (2025)

| Crate | Purpose |
|-------|---------|
| `crossbeam` | Lock-free data structures, scoped threads, channels |
| `crossbeam-channel` | High-performance MPMC channels |
| `crossbeam-epoch` | Epoch-based memory reclamation for lock-free structures |
| `parking_lot` | Faster Mutex/RwLock replacements (smaller, no poisoning) |
| `rayon` | Data parallelism (parallel iterators) |
| `tokio` | Async runtime with timers, IO, channels |
| `flume` | Fast MPMC channel (async + sync) |
| `arc-swap` | Atomically swappable `Arc` for read-heavy concurrent access |
| `dashmap` | Concurrent HashMap (sharded locking) |

### 10.8 Cargo Scripts (Unstable, Coming Soon)

Single-file Rust scripts with inline dependency declarations:
```rust
#!/usr/bin/env cargo
---
[dependencies]
serde = { version = "1", features = ["derive"] }
---

fn main() {
    println!("Hello from a cargo script!");
}
```

Run with `cargo +nightly -Zscript script.rs`. Being tracked for stabilization.

---

## Appendix B: Study Path

| Week | Focus | Exercises |
|------|-------|-----------|
| 1-2 | Parts 1-2: Memory, Ownership, Borrowing | Ex 1.1-1.3, 2.1-2.4 |
| 3-4 | Parts 3-4: Lifetimes, Types, Traits | Ex 3.1-3.3 (StrSplit), 4.1-4.4 |
| 5-6 | Part 5: Concurrency | Ex 5.1-5.2, Thread Pool project |
| 7-8 | Parts 6-7: Atomics, Memory Ordering | Ex 6.1-6.2, 7.1-7.2 (OneShotChannel) |
| 9-10 | Part 8: Primitives + Capstones | SpinLock, Lazy, MPSC channel, Lock-free stack |
| Ongoing | Read real code | `crossbeam`, `parking_lot`, `tokio` source |

**Daily practice:** Write code until borrow checker errors feel intuitive. Run everything under `cargo +nightly miri test`. Read other people's Rust — the standard library source is excellent.
