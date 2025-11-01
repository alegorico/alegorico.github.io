---
layout: post
title: Java Funcional - Transformando el Paradigma de Programación
tags: [java, functional-programming, lambda, streams, pattern-matching, records]
---

Java ha evolucionado dramáticamente desde Java 8, abrazando paradigmas funcionales que han transformado la forma de escribir código. Mi experiencia trabajando con proyectos como [java-design-patterns-base](https://github.com/alegorico/java-design-patterns-base) me ha mostrado cómo estos cambios han revolucionado el desarrollo empresarial.

## La Revolución Funcional en Java

### Del Imperativo al Declarativo

```java
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

// ❌ Estilo imperativo tradicional
public List<String> processCustomersOldWay(List<Customer> customers) {
    List<Customer> activeCustomers = new ArrayList<>();
    
    // Filtrar clientes activos
    for (Customer customer : customers) {
        if (customer.isActive() && customer.getOrderCount() > 5) {
            activeCustomers.add(customer);
        }
    }
    
    // Ordenar por nombre
    Collections.sort(activeCustomers, new Comparator<Customer>() {
        @Override
        public int compare(Customer c1, Customer c2) {
            return c1.getName().compareTo(c2.getName());
        }
    });
    
    // Extraer nombres
    List<String> names = new ArrayList<>();
    for (Customer customer : activeCustomers) {
        names.add(customer.getName().toUpperCase());
    }
    
    return names;
}

// ✅ Estilo funcional moderno
public List<String> processCustomersFunctional(List<Customer> customers) {
    return customers.stream()
            .filter(Customer::isActive)
            .filter(c -> c.getOrderCount() > 5)
            .sorted(Comparator.comparing(Customer::getName))
            .map(Customer::getName)
            .map(String::toUpperCase)
            .collect(Collectors.toList());
}
```

### Mi Experiencia con Design Patterns Funcionales

En el proyecto [java-design-patterns-base](https://github.com/alegorico/java-design-patterns-base), he explorado cómo los patrones clásicos se transforman con programación funcional:

```java
// Strategy Pattern funcional
@FunctionalInterface
public interface PricingStrategy {
    BigDecimal calculatePrice(BigDecimal basePrice, Customer customer);
}

public class PricingService {
    // Estrategias como funciones
    private static final Map<CustomerType, PricingStrategy> STRATEGIES = Map.of(
        CustomerType.REGULAR, (price, customer) -> price,
        CustomerType.PREMIUM, (price, customer) -> price.multiply(new BigDecimal("0.9")),
        CustomerType.VIP, (price, customer) -> price.multiply(new BigDecimal("0.8")),
        CustomerType.ENTERPRISE, (price, customer) -> 
            customer.getAnnualVolume().compareTo(new BigDecimal("100000")) > 0 
                ? price.multiply(new BigDecimal("0.7")) 
                : price.multiply(new BigDecimal("0.85"))
    );
    
    public BigDecimal calculatePrice(BigDecimal basePrice, Customer customer) {
        return STRATEGIES.getOrDefault(customer.getType(), STRATEGIES.get(CustomerType.REGULAR))
                        .calculatePrice(basePrice, customer);
    }
}

// Command Pattern con funciones
public class OrderProcessor {
    private final List<Function<Order, Order>> processors = new ArrayList<>();
    
    public OrderProcessor addProcessor(Function<Order, Order> processor) {
        processors.add(processor);
        return this;
    }
    
    public Order processOrder(Order order) {
        return processors.stream()
                .reduce(Function.identity(), Function::andThen)
                .apply(order);
    }
}

// Uso del processor funcional
var processor = new OrderProcessor()
    .addProcessor(order -> order.withTotalCalculated())
    .addProcessor(order -> order.withTaxesApplied())
    .addProcessor(order -> order.withDiscountsApplied())
    .addProcessor(order -> order.withInventoryReserved());

Order processedOrder = processor.processOrder(originalOrder);
```

## Características Funcionales Avanzadas

### 1. Streams y Collectors Personalizados

```java
import java.util.stream.Collector;

// Collector personalizado para estadísticas
public static Collector<Order, ?, OrderStatistics> orderStatistics() {
    return Collector.of(
        OrderStatistics::new,
        (stats, order) -> {
            stats.addOrder(order.getTotal());
            stats.incrementCount();
            if (order.isPremium()) stats.incrementPremiumCount();
        },
        (stats1, stats2) -> {
            stats1.merge(stats2);
            return stats1;
        },
        Collector.Characteristics.UNORDERED
    );
}

// Uso avanzado de Streams
public class SalesAnalyzer {
    
    public Map<String, Double> calculateMonthlyRevenue(List<Order> orders) {
        return orders.stream()
                .filter(order -> order.getStatus() == OrderStatus.COMPLETED)
                .collect(Collectors.groupingBy(
                    order -> order.getDate().format(DateTimeFormatter.ofPattern("yyyy-MM")),
                    Collectors.summingDouble(order -> order.getTotal().doubleValue())
                ));
    }
    
    public Optional<Customer> findTopCustomer(List<Order> orders) {
        return orders.stream()
                .collect(Collectors.groupingBy(
                    Order::getCustomer,
                    Collectors.summingDouble(order -> order.getTotal().doubleValue())
                ))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey);
    }
    
    // Procesamiento paralelo para grandes volúmenes
    public CompletableFuture<SalesReport> generateReportAsync(List<Order> orders) {
        return CompletableFuture.supplyAsync(() -> {
            var stats = orders.parallelStream()
                    .filter(order -> order.getDate().isAfter(LocalDate.now().minusMonths(12)))
                    .collect(orderStatistics());
            
            var topProducts = orders.parallelStream()
                    .flatMap(order -> order.getItems().stream())
                    .collect(Collectors.groupingBy(
                        OrderItem::getProductName,
                        Collectors.summingInt(OrderItem::getQuantity)
                    ))
                    .entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(10)
                    .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                    ));
            
            return new SalesReport(stats, topProducts);
        });
    }
}
```

### 2. Optional y Manejo Funcional de Nulls

```java
public class CustomerService {
    
    // Encadenamiento seguro con Optional
    public Optional<String> getCustomerEmailDomain(Long customerId) {
        return findCustomerById(customerId)
                .map(Customer::getEmail)
                .filter(email -> email.contains("@"))
                .map(email -> email.substring(email.indexOf("@") + 1))
                .map(String::toLowerCase);
    }
    
    // Combinación de múltiples Optional
    public Optional<CustomerProfile> buildCompleteProfile(Long customerId) {
        var customerOpt = findCustomerById(customerId);
        var preferencesOpt = findCustomerPreferences(customerId);
        var historyOpt = findOrderHistory(customerId);
        
        return customerOpt.flatMap(customer ->
            preferencesOpt.flatMap(preferences ->
                historyOpt.map(history ->
                    new CustomerProfile(customer, preferences, history)
                )
            )
        );
    }
    
    // Alternativas funcionales con OrElse
    public Customer getCustomerWithDefaults(Long customerId) {
        return findCustomerById(customerId)
                .filter(Customer::isActive)
                .or(() -> findArchivedCustomer(customerId))
                .orElseGet(() -> Customer.createGuest("Unknown Customer"));
    }
    
    // Transformación condicional
    public Optional<BigDecimal> calculateDiscount(Customer customer, Order order) {
        return Optional.of(customer)
                .filter(c -> c.getType() != CustomerType.REGULAR)
                .filter(c -> order.getTotal().compareTo(new BigDecimal("100")) > 0)
                .map(c -> switch (c.getType()) {
                    case PREMIUM -> order.getTotal().multiply(new BigDecimal("0.05"));
                    case VIP -> order.getTotal().multiply(new BigDecimal("0.10"));
                    case ENTERPRISE -> order.getTotal().multiply(new BigDecimal("0.15"));
                    default -> BigDecimal.ZERO;
                });
    }
}
```

### 3. Functional Interfaces y Method References

```java
// Interfaces funcionales personalizadas
@FunctionalInterface
public interface TriFunction<T, U, V, R> {
    R apply(T t, U u, V v);
    
    default <W> TriFunction<T, U, V, W> andThen(Function<? super R, ? extends W> after) {
        Objects.requireNonNull(after);
        return (T t, U u, V v) -> after.apply(apply(t, u, v));
    }
}

@FunctionalInterface
public interface ValidatorFunction<T> {
    ValidationResult validate(T object);
    
    default ValidatorFunction<T> and(ValidatorFunction<T> other) {
        return object -> {
            ValidationResult first = this.validate(object);
            if (!first.isValid()) return first;
            return other.validate(object);
        };
    }
}

// Uso avanzado de method references
public class ValidationService {
    
    private final Map<Class<?>, List<ValidatorFunction<?>>> validators = new HashMap<>();
    
    @SuppressWarnings("unchecked")
    public <T> ValidationResult validate(T object) {
        Class<?> clazz = object.getClass();
        
        return validators.getOrDefault(clazz, Collections.emptyList())
                .stream()
                .map(validator -> ((ValidatorFunction<T>) validator).validate(object))
                .filter(result -> !result.isValid())
                .findFirst()
                .orElse(ValidationResult.valid());
    }
    
    // Configuración fluida con method references
    public ValidationService configure() {
        // Customer validators
        addValidator(Customer.class, Customer::validateEmail);
        addValidator(Customer.class, Customer::validatePhoneNumber);
        addValidator(Customer.class, this::validateCustomerAge);
        
        // Order validators  
        addValidator(Order.class, Order::validateItems);
        addValidator(Order.class, this::validateOrderTotal);
        addValidator(Order.class, this::validateDeliveryAddress);
        
        return this;
    }
    
    private <T> void addValidator(Class<T> clazz, ValidatorFunction<T> validator) {
        validators.computeIfAbsent(clazz, k -> new ArrayList<>()).add(validator);
    }
}
```

## Características Modernas de Java

### 1. Records y Pattern Matching (Java 14+)

```java
// Records para DTOs inmutables
public record CustomerSummary(
    String name,
    String email,
    CustomerType type,
    BigDecimal totalSpent,
    LocalDate lastOrderDate
) {
    // Validación en constructor compacto
    public CustomerSummary {
        Objects.requireNonNull(name, "Name cannot be null");
        Objects.requireNonNull(email, "Email cannot be null");
        if (totalSpent.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Total spent cannot be negative");
        }
    }
    
    // Métodos derivados
    public boolean isRecentCustomer() {
        return lastOrderDate.isAfter(LocalDate.now().minusMonths(3));
    }
    
    public String getFormattedSpent() {
        return NumberFormat.getCurrencyInstance().format(totalSpent);
    }
}

// Pattern matching con switch expressions
public String processPayment(Payment payment) {
    return switch (payment) {
        case CreditCardPayment(var cardNumber, var cvv, var amount) -> 
            processCreditCard(cardNumber, cvv, amount);
            
        case PayPalPayment(var email, var amount) -> 
            processPayPal(email, amount);
            
        case BankTransferPayment(var accountNumber, var routingNumber, var amount) -> 
            processBankTransfer(accountNumber, routingNumber, amount);
            
        case CryptocurrencyPayment(var walletAddress, var currency, var amount) -> 
            processCrypto(walletAddress, currency, amount);
    };
}

// Sealed classes para jerarquías controladas
public sealed interface Payment 
    permits CreditCardPayment, PayPalPayment, BankTransferPayment, CryptocurrencyPayment {
    
    BigDecimal getAmount();
    
    default PaymentCategory getCategory() {
        return switch (this) {
            case CreditCardPayment cc -> PaymentCategory.TRADITIONAL;
            case PayPalPayment pp -> PaymentCategory.DIGITAL;
            case BankTransferPayment bt -> PaymentCategory.TRADITIONAL;
            case CryptocurrencyPayment cp -> PaymentCategory.CRYPTOCURRENCY;
        };
    }
}
```

### 2. CompletableFuture y Programación Reactiva

```java
public class AsyncOrderProcessor {
    
    private final ExecutorService executorService = 
        ForkJoinPool.commonPool();
    
    // Procesamiento asíncrono en pipeline
    public CompletableFuture<OrderResult> processOrderAsync(Order order) {
        return CompletableFuture
            .supplyAsync(() -> validateOrder(order), executorService)
            .thenCompose(validOrder -> 
                CompletableFuture.allOf(
                    reserveInventoryAsync(validOrder),
                    calculateShippingAsync(validOrder),
                    processPaymentAsync(validOrder)
                ).thenApply(void_ -> validOrder)
            )
            .thenCompose(this::finalizeOrderAsync)
            .thenApply(finalizedOrder -> new OrderResult(finalizedOrder, "SUCCESS"))
            .exceptionally(throwable -> {
                log.error("Error processing order: " + order.getId(), throwable);
                return new OrderResult(order, "FAILED: " + throwable.getMessage());
            });
    }
    
    // Procesamiento de lotes con límite de concurrencia
    public CompletableFuture<List<OrderResult>> processBatchAsync(
            List<Order> orders, int maxConcurrency) {
        
        Semaphore semaphore = new Semaphore(maxConcurrency);
        
        List<CompletableFuture<OrderResult>> futures = orders.stream()
            .map(order -> CompletableFuture
                .runAsync(() -> {
                    try {
                        semaphore.acquire();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(e);
                    }
                })
                .thenCompose(void_ -> processOrderAsync(order))
                .whenComplete((result, throwable) -> semaphore.release())
            )
            .collect(Collectors.toList());
        
        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenApply(void_ -> futures.stream()
                .map(CompletableFuture::join)
                .collect(Collectors.toList())
            );
    }
    
    // Timeout y fallback patterns
    public CompletableFuture<String> getRecommendationsWithFallback(Long customerId) {
        var primaryService = getRecommendationsFromAI(customerId)
            .orTimeout(2, TimeUnit.SECONDS);
            
        var fallbackService = getRecommendationsFromHistory(customerId)
            .orTimeout(5, TimeUnit.SECONDS);
        
        return primaryService
            .exceptionally(throwable -> null)
            .thenCompose(result -> 
                result != null 
                    ? CompletableFuture.completedFuture(result)
                    : fallbackService
            )
            .exceptionally(throwable -> "No recommendations available");
    }
}
```

### 3. Functional Error Handling

```java
// Result type para manejo funcional de errores
public sealed interface Result<T, E> {
    
    record Success<T, E>(T value) implements Result<T, E> {}
    record Failure<T, E>(E error) implements Result<T, E> {}
    
    static <T, E> Result<T, E> success(T value) {
        return new Success<>(value);
    }
    
    static <T, E> Result<T, E> failure(E error) {
        return new Failure<>(error);
    }
    
    default <U> Result<U, E> map(Function<T, U> mapper) {
        return switch (this) {
            case Success<T, E>(var value) -> success(mapper.apply(value));
            case Failure<T, E>(var error) -> failure(error);
        };
    }
    
    default <U> Result<U, E> flatMap(Function<T, Result<U, E>> mapper) {
        return switch (this) {
            case Success<T, E>(var value) -> mapper.apply(value);
            case Failure<T, E>(var error) -> failure(error);
        };
    }
    
    default Result<T, E> filter(Predicate<T> predicate, E errorOnFalse) {
        return switch (this) {
            case Success<T, E>(var value) -> 
                predicate.test(value) ? this : failure(errorOnFalse);
            case Failure<T, E> failure -> failure;
        };
    }
    
    default T orElse(T defaultValue) {
        return switch (this) {
            case Success<T, E>(var value) -> value;
            case Failure<T, E>(var error) -> defaultValue;
        };
    }
}

// Uso del Result type
public class CustomerService {
    
    public Result<Customer, String> createCustomer(CustomerRequest request) {
        return validateCustomerRequest(request)
            .flatMap(this::checkDuplicateEmail)
            .flatMap(this::saveCustomer)
            .map(this::enrichCustomerData);
    }
    
    private Result<CustomerRequest, String> validateCustomerRequest(CustomerRequest request) {
        if (request.email() == null || !request.email().contains("@")) {
            return Result.failure("Invalid email address");
        }
        if (request.name() == null || request.name().trim().isEmpty()) {
            return Result.failure("Name is required");
        }
        return Result.success(request);
    }
    
    // Composición funcional de operaciones que pueden fallar
    public Result<OrderSummary, String> calculateOrderSummary(Long orderId) {
        return findOrder(orderId)
            .flatMap(this::validateOrderStatus)
            .flatMap(order -> 
                Result.success(order)
                    .flatMap(this::calculateSubtotal)
                    .flatMap(subtotal -> calculateTax(order, subtotal))
                    .flatMap(tax -> calculateShipping(order).map(shipping -> 
                        new OrderSummary(order.getId(), subtotal, tax, shipping)
                    ))
            );
    }
}
```

## Performance y Optimización

### 1. Lazy Evaluation y Streams Infinitos

```java
public class DataProcessor {
    
    // Stream infinito con lazy evaluation
    public Stream<BigInteger> fibonacciSequence() {
        return Stream.iterate(
            new BigInteger[]{BigInteger.ZERO, BigInteger.ONE},
            pair -> new BigInteger[]{pair[1], pair[0].add(pair[1])}
        ).map(pair -> pair[0]);
    }
    
    // Procesamiento lazy de grandes datasets
    public Optional<Customer> findHighValueCustomer(String filePath) {
        try (Stream<String> lines = Files.lines(Paths.get(filePath))) {
            return lines
                .skip(1) // Skip header
                .map(this::parseCustomerLine)
                .filter(Objects::nonNull)
                .filter(customer -> customer.getTotalSpent().compareTo(new BigDecimal("10000")) > 0)
                .filter(Customer::isActive)
                .findFirst(); // Short-circuit - solo procesa hasta encontrar el primero
        } catch (IOException e) {
            log.error("Error reading customer file", e);
            return Optional.empty();
        }
    }
    
    // Memoización funcional
    private final Map<String, Function<BigDecimal, BigDecimal>> memoizedTaxCalculators = 
        new ConcurrentHashMap<>();
    
    public Function<BigDecimal, BigDecimal> getTaxCalculator(String region) {
        return memoizedTaxCalculators.computeIfAbsent(region, this::createTaxCalculator);
    }
    
    private Function<BigDecimal, BigDecimal> createTaxCalculator(String region) {
        BigDecimal taxRate = TaxRateService.getRateForRegion(region);
        return amount -> amount.multiply(taxRate);
    }
}
```

### 2. Collectors Paralelos y Custom Spliterators

```java
public class AdvancedCollectors {
    
    // Collector paralelo para operaciones costosas
    public static <T> Collector<T, ?, Map<Boolean, List<T>>> 
            partitioningByAsync(Function<T, CompletableFuture<Boolean>> asyncPredicate) {
        
        return Collector.of(
            () -> new ConcurrentHashMap<Boolean, List<T>>(),
            (map, item) -> {
                CompletableFuture<Boolean> future = asyncPredicate.apply(item);
                Boolean result = future.join(); // En producción, usar timeout
                map.computeIfAbsent(result, k -> new ArrayList<>()).add(item);
            },
            (map1, map2) -> {
                map2.forEach((key, value) -> 
                    map1.merge(key, value, (list1, list2) -> {
                        list1.addAll(list2);
                        return list1;
                    })
                );
                return map1;
            }
        );
    }
    
    // Procesamiento de grandes volúmenes con spliterator personalizado
    public static class BatchSpliterator<T> implements Spliterator<List<T>> {
        private final Spliterator<T> source;
        private final int batchSize;
        
        public BatchSpliterator(Spliterator<T> source, int batchSize) {
            this.source = source;
            this.batchSize = batchSize;
        }
        
        @Override
        public boolean tryAdvance(Consumer<? super List<T>> action) {
            List<T> batch = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize && source.tryAdvance(batch::add); i++);
            
            if (!batch.isEmpty()) {
                action.accept(batch);
                return true;
            }
            return false;
        }
        
        @Override
        public Spliterator<List<T>> trySplit() {
            Spliterator<T> split = source.trySplit();
            return split == null ? null : new BatchSpliterator<>(split, batchSize);
        }
        
        @Override
        public long estimateSize() {
            return (source.estimateSize() + batchSize - 1) / batchSize;
        }
        
        @Override
        public int characteristics() {
            return source.characteristics();
        }
    }
}
```

## Casos de Uso Empresariales

### 1. Pipeline de Datos Funcional

```java
public class SalesDataPipeline {
    
    public CompletableFuture<SalesReport> processSalesData(LocalDate from, LocalDate to) {
        return CompletableFuture
            .supplyAsync(() -> loadRawSalesData(from, to))
            .thenApply(this::cleanAndValidateData)
            .thenApply(this::enrichWithCustomerData)
            .thenApply(this::calculateMetrics)
            .thenCompose(this::generateInsightsAsync)
            .thenApply(this::formatReport)
            .exceptionally(this::handleError);
    }
    
    private List<SalesRecord> cleanAndValidateData(List<RawSalesRecord> rawData) {
        return rawData.parallelStream()
            .filter(record -> record.getAmount().compareTo(BigDecimal.ZERO) > 0)
            .filter(record -> record.getDate().isAfter(LocalDate.of(2020, 1, 1)))
            .map(this::normalizeRecord)
            .filter(Optional::isPresent)
            .map(Optional::get)
            .collect(Collectors.toList());
    }
    
    // Enriquecimiento de datos con cache funcional
    private final Function<String, Optional<Customer>> cachedCustomerLookup = 
        Caffeine.newBuilder()
            .maximumSize(10_000)
            .expireAfterWrite(Duration.ofMinutes(30))
            .build()
            .asMap()
            .computeIfAbsent(this::loadCustomer);
    
    private List<EnrichedSalesRecord> enrichWithCustomerData(List<SalesRecord> records) {
        return records.parallelStream()
            .map(record -> cachedCustomerLookup.apply(record.getCustomerId())
                .map(customer -> new EnrichedSalesRecord(record, customer))
                .orElse(new EnrichedSalesRecord(record, Customer.unknown())))
            .collect(Collectors.toList());
    }
}
```

### 2. Sistema de Reglas Funcional

```java
public class BusinessRulesEngine {
    
    @FunctionalInterface
    public interface BusinessRule<T> {
        RuleResult evaluate(T context);
        
        default BusinessRule<T> and(BusinessRule<T> other) {
            return context -> {
                RuleResult first = this.evaluate(context);
                if (!first.passed()) return first;
                return other.evaluate(context);
            };
        }
        
        default BusinessRule<T> or(BusinessRule<T> other) {
            return context -> {
                RuleResult first = this.evaluate(context);
                if (first.passed()) return first;
                return other.evaluate(context);
            };
        }
    }
    
    // Configuración declarativa de reglas
    public class LoanApprovalEngine {
        private final BusinessRule<LoanApplication> approvalRules;
        
        public LoanApprovalEngine() {
            this.approvalRules = createCreditScoreRule()
                .and(createIncomeRule())
                .and(createDebtToIncomeRule())
                .and(createEmploymentHistoryRule());
        }
        
        private BusinessRule<LoanApplication> createCreditScoreRule() {
            return application -> 
                application.getCreditScore() >= 650
                    ? RuleResult.passed()
                    : RuleResult.failed("Credit score too low: " + application.getCreditScore());
        }
        
        private BusinessRule<LoanApplication> createIncomeRule() {
            return application -> {
                BigDecimal minIncome = application.getLoanAmount().divide(new BigDecimal("5"));
                return application.getAnnualIncome().compareTo(minIncome) >= 0
                    ? RuleResult.passed()
                    : RuleResult.failed("Insufficient income");
            };
        }
        
        public LoanDecision evaluateApplication(LoanApplication application) {
            RuleResult result = approvalRules.evaluate(application);
            
            return new LoanDecision(
                application.getId(),
                result.passed() ? LoanStatus.APPROVED : LoanStatus.REJECTED,
                result.getMessage(),
                calculateInterestRate(application, result)
            );
        }
        
        private BigDecimal calculateInterestRate(LoanApplication app, RuleResult result) {
            if (!result.passed()) return BigDecimal.ZERO;
            
            return Stream.of(
                    assessCreditScoreRisk(app.getCreditScore()),
                    assessIncomeStabilityRisk(app.getEmploymentHistory()),
                    assessLoanAmountRisk(app.getLoanAmount())
                )
                .reduce(new BigDecimal("3.5"), BigDecimal::add) // Base rate
                .min(new BigDecimal("15.0")); // Cap at 15%
        }
    }
}
```

## Integración con Herramientas Modernas

Mi trabajo con [java-design-patterns-base](https://github.com/alegorico/java-design-patterns-base) me ha mostrado la importancia de integrar paradigmas funcionales en arquitecturas empresariales:

```java
// Microservicios funcionales con Spring WebFlux
@RestController
public class FunctionalCustomerController {
    
    private final CustomerService customerService;
    
    @GetMapping("/customers/{id}/orders")
    public Mono<ResponseEntity<List<OrderSummary>>> getCustomerOrders(
            @PathVariable Long id,
            @RequestParam Optional<LocalDate> from,
            @RequestParam Optional<LocalDate> to) {
        
        return customerService.findById(id)
            .switchIfEmpty(Mono.error(new CustomerNotFoundException(id)))
            .flatMap(customer -> 
                customerService.getOrders(
                    customer.getId(), 
                    from.orElse(LocalDate.now().minusYears(1)),
                    to.orElse(LocalDate.now())
                )
            )
            .map(orders -> orders.stream()
                .map(OrderSummary::from)
                .collect(Collectors.toList())
            )
            .map(ResponseEntity::ok)
            .onErrorMap(CustomerNotFoundException.class, 
                ex -> new ResponseStatusException(HttpStatus.NOT_FOUND, ex.getMessage())
            );
    }
    
    // Composición funcional de validaciones
    @PostMapping("/customers")
    public Mono<ResponseEntity<Customer>> createCustomer(@RequestBody CustomerRequest request) {
        return Mono.just(request)
            .flatMap(this::validateRequest)
            .flatMap(customerService::create)
            .map(customer -> ResponseEntity.status(HttpStatus.CREATED).body(customer))
            .onErrorMap(ValidationException.class,
                ex -> new ResponseStatusException(HttpStatus.BAD_REQUEST, ex.getMessage())
            );
    }
    
    private Mono<CustomerRequest> validateRequest(CustomerRequest request) {
        return Mono.fromSupplier(() -> 
            Stream.of(
                validateEmail(request.email()),
                validateName(request.name()),
                validatePhone(request.phone())
            )
            .filter(result -> !result.isValid())
            .findFirst()
            .map(result -> {
                throw new ValidationException(result.getMessage());
            })
            .orElse(request)
        );
    }
}
```

## Conclusiones

La programación funcional en Java ha transformado el desarrollo empresarial:

🎯 **Código más expresivo**: Declarativo vs imperativo  
⚡ **Mejor composición**: Funciones como building blocks  
🔒 **Inmutabilidad**: Records y datos inmutables por defecto  
🚀 **Paralelización**: Streams paralelos y async programming  
🛡️ **Error handling**: Tipos de resultado y Optional  
🔧 **Testabilidad**: Funciones puras más fáciles de testear  

Mi experiencia con `java-design-patterns-base` confirma que estos paradigmas no solo mejoran la calidad del código, sino que facilitan el mantenimiento y la evolución de sistemas complejos.

---
*¿Has adoptado programación funcional en tus proyectos Java? Comparte tu experiencia y desafíos en los comentarios.*