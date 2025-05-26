# GitHub Copilot Instructions: Swift Coding Style & Best Practices

## I. General Principles

*   **Clarity First:** Code should be easy to read and understand. Prioritize self-documenting code over excessive comments.
*   **Conciseness:** Write expressive but not overly verbose code.
*   **Safety:** Leverage Swift's type safety and error handling features. Avoid runtime crashes where possible.
*   **Performance Awareness:** Write efficient code, but don't prematurely optimize. Profile before optimizing.
*   **Modularity:** Design components with clear responsibilities and well-defined interfaces.

## II. Swift Language Best Practices (Swift 6.0+)

1.  **Type Safety & Inference:**
    *   Leverage type inference but provide explicit types for public APIs or where clarity is improved.
        ```swift
        // Good (inference is clear)
        let count = 5
        let message = "Hello"
        let scores = [10.0, 9.5, 8.0]

        // Good (explicit for clarity or API boundary)
        public func processData(_ data: Data) -> String { ... }
        let complexResult: Result<Int, MyError> = someFunction()
        ```
    *   Use non-optional types by default. Only use optionals (`?`, `!`) when a value can genuinely be absent.
    *   Avoid force unwrapping (`!`). Use `guard let`, `if let`, optional chaining (`?.`), and the nil-coalescing operator (`??`).
        ```swift
        // Bad
        // let name = optionalName!

        // Good
        guard let name = optionalName else {
            print("Name is missing.")
            return
        }
        print("Hello, \(name)")

        // Good
        let streetName = user.address?.street?.name ?? "N/A"
        ```

2.  **Immutability:**
    *   Prefer `let` for constants over `var` for variables to promote safer code.
        ```swift
        // Good
        let maximumAttempts = 3
        var currentAttempt = 0

        // Avoid if value doesn't change after init
        // var configuration = AppConfig()
        ```
    *   Favor immutable collections (`Array`, `Dictionary`, `Set` declared with `let`) when the collection itself won't be modified after creation.

3.  **Value vs. Reference Types:**
    *   Prefer `struct` and `enum` (value types) over `class` (reference type) for data modeling unless class-specific features like identity, inheritance, or deinitializers are required.
        ```swift
        // Good for data models
        struct UserProfile {
            let id: UUID
            var username: String
        }

        enum NetworkState {
            case loading
            case success(Data)
            case failure(Error)
        }
        ```

4.  **Error Handling:**
    *   Use Swift's `Error` protocol and `throw`/`try`/`catch` for recoverable errors.
    *   Define custom error types (enums are great for this) for better context.
        ```swift
        enum DataProcessingError: Error, LocalizedError {
            case invalidFormat
            case missingRequiredField(String)

            var errorDescription: String? {
                switch self {
                case .invalidFormat: return "The data format is invalid."
                case .missingRequiredField(let field): return "Missing required field: \(field)."
                }
            }
        }

        func parse(data: Data) throws -> ParsedObject {
            guard isValid(data) else { throw DataProcessingError.invalidFormat }
            // ... parsing logic
        }

        do {
            let object = try parse(data: myData)
        } catch let error as DataProcessingError {
            print("Processing failed: \(error.localizedDescription)")
        } catch {
            print("An unexpected error occurred: \(error)")
        }
        ```
    *   Avoid using `try!` unless you are certain an operation will never throw an error at runtime (rare).

5.  **Concurrency (async/await):**
    *   Use `async/await` for asynchronous operations.
    *   Mark functions performing asynchronous work with `async` and call them with `await`.
    *   Ensure `Sendable` conformance for types passed across actor boundaries or used in `Task` initializers.
        ```swift
        struct Item: Sendable { // if used across actor boundaries
            let id: String
        }

        actor DataStore {
            private var items: [String: Item] = [:]

            func fetchItem(id: String) async -> Item? {
                // Simulate network delay
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                return items[id]
            }

            func storeItem(_ item: Item) {
                items[item.id] = item
            }
        }

        func loadAndProcessItem(id: String, store: DataStore) async {
            if let item = await store.fetchItem(id: id) {
                print("Fetched item: \(item.id)")
            } else {
                print("Item not found.")
            }
        }
        ```
    *   Use `Task { ... }` for creating new unstructured tasks.
    *   Use structured concurrency (e.g., `async let`, task groups) where appropriate.
    *   For UI updates resulting from background work, dispatch back to the `@MainActor`.
        ```swift
        // @MainActor
        // class ViewModel: ObservableObject {
        //     @Published var data: String = "Loading..."
        //
        //     func fetchData() {
        //         Task {
        //             let result = await someBackgroundWork()
        //             await MainActor.run {
        //                 self.data = result
        //             }
        //         }
        //     }
        // }
        ```

6.  **Control Flow:**
    *   Use `guard` for early exits to improve readability by reducing nesting.
        ```swift
        func process(user: User?) {
            guard let validUser = user else {
                print("User is nil, cannot process.")
                return
            }
            guard validUser.isActive else {
                print("User is not active.")
                return
            }
            // Process active user
        }
        ```
    *   Use `for-in` loops for iteration. Use `forEach` for simple operations where the index isn't needed and `break/continue` aren't required.
    *   `switch` statements should be exhaustive. Use `default` sparingly, preferring to list all cases.

7.  **Functions and Methods:**
    *   Use clear, descriptive names that follow Swift API Design Guidelines (e.g., name functions for their side effects, use prepositions for parameters).
        ```swift
        // Good
        func processPayment(for order: Order, usingMethod paymentMethod: PaymentMethod) throws -> PaymentReceipt
        user.updateName(to: "New Name")
        let filteredItems = items.filter { $0.isAvailable }
        ```
    *   Keep functions focused on a single responsibility.

8.  **Access Control:**
    *   Use the most restrictive access level that makes sense (`private`, `fileprivate`, `internal`, `public`, `open`). Default to `internal`.
    *   `private` restricts visibility to the enclosing declaration.
    *   `fileprivate` restricts visibility to the current file.

9.  **Code Formatting and Style:**
    *   **Indentation:** Use 4 spaces for indentation.
    *   **Line Length:** Aim for lines no longer than 100-120 characters where practical for readability.
    *   **Whitespace:** Use whitespace to improve readability (e.g., around operators, after commas, blank lines between logical blocks of code).
    *   **Braces:** Opening braces for control statements, functions, classes, etc., should generally be on the same line.
        ```swift
        if condition {
            // do something
        } else {
            // do something else
        }

        func myFunction() {
            // ...
        }
        ```

10. **Naming Conventions:**
    *   Use UpperCamelCase for type names (classes, structs, enums, protocols).
    *   Use lowerCamelCase for variable names, function parameters, and function/method names.
        ```swift
        struct UserProfileManager { ... }
        var userName: String
        func fetchUserDetails() { ... }
        ```

11. **Comments:**
    *   **No Redundant Comments:** Avoid comments that explain *what* the code is doing if the code is already clear.
        ```swift
        // Bad:
        // Increment the counter
        // counter += 1

        // Good: Code is self-explanatory
        counter += 1
        ```
    *   **Explain *Why*, Not *What*:** Use comments to explain non-obvious logic, complex algorithms, or the reasoning behind a particular design choice.
        ```swift
        // Good: Explains a non-obvious workaround
        // APIv1 has a bug where it returns an empty array instead of nil for no results.
        // We handle this by checking isEmpty before proceeding.
        if results.isEmpty { return }
        ```
    *   **DocC Comments (`///`):** Use DocC comments for public APIs to generate documentation.
        ```swift
        /// Processes user input and returns a validated response.
        ///
        /// - Parameter input: The raw user input string.
        /// - Throws: `InputError.invalidCharacters` if input contains disallowed characters.
        /// - Returns: A validated string.
        public func processInput(_ input: String) throws -> String { ... }
        ```
    *   **MARK, FIXME, TODO:** Use these for organizing code and noting work items.
        ```swift
        // MARK: - User Authentication

        // FIXME: This algorithm has O(n^2) complexity, needs optimization.

        // TODO: Add support for OAuth2.
        ```

12. **API Design (General):**
    *   Strive for clarity at the point of use.
    *   Name functions and methods according to their side-effects.
    *   Use argument labels wisely to improve readability.

13. **Testing:**
    *   Run tests using the following command:
        ```bash
        xcodebuild test -scheme mlx_vlm -destination 'platform=OS X'
        ```
    *   Write comprehensive unit tests for public APIs.
    *   Follow the Arrange-Act-Assert pattern for test structure.
    *   Test both happy paths and edge cases.

This guide provides a solid foundation. The Swift API Design Guidelines are the ultimate reference. Always prioritize making the code understandable for your future self and your team.