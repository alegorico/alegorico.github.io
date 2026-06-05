---
layout: post
title: Java 21 LTS - Características Revolucionarias para el Desarrollo Moderno
tags: [java, java21, lts, virtual-threads, pattern-matching, records, sealed-classes]
---

Java 21 LTS marca un hito en la evolución del lenguaje, introduciendo características que transforman radicalmente cómo desarrollamos aplicaciones. Mi experiencia migrando proyectos como [java-design-patterns-base](https://github.com/alegorico/java-design-patterns-base) a estas nuevas características me ha mostrado el impacto revolucionario de estas innovaciones.

## Virtual Threads: Concurrencia Sin Límites

### El Fin del Thread Pool Hell

```java
import java.util.concurrent.Executors;
import java.time.Duration;

public class VirtualThreadsDemo {
    
    // ❌ Threads tradicionales (limitados)
    public void traditionalThreadsApproach() throws InterruptedException {
        try (var executor = Executors.newFixedThreadPool(200)) {
            for (int i = 0; i < 10_000; i++) {
                final int taskId = i;
                executor.submit(() -> {
                    try {
                        // Simular I/O blocking
                        Thread.sleep(Duration.ofSeconds(1));
                        System.out.println("Traditional task " + taskId + " completed");
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
        } // Esto agotaría los threads del pool
    }
    
    // ✅ Virtual Threads (escalables)
    public void virtualThreadsApproach() throws InterruptedException {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 10_000; i++) {
                final int taskId = i;
                executor.submit(() -> {
                    try {
                        // Mismo I/O blocking, pero eficiente
                        Thread.sleep(Duration.ofSeconds(1));
                        System.out.println("Virtual task " + taskId + " completed");
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }
        } // 10,000 virtual threads ¡sin problema!
    }
    
    // Server HTTP con virtual threads
    public void startVirtualThreadServer() throws IOException {
        var server = HttpServer.create(new InetSocketAddress(8080), 0);
        
        server.createContext("/api/process", exchange -> {
            // Cada request maneja en su propio virtual thread
            Thread.startVirtualThread(() -> {
                try {
                    handleRequest(exchange);
                } catch (Exception e) {
                    handleError(exchange, e);
                }
            });
        });
        
        server.start();
        System.out.println("Server started with virtual threads");
    }
    
    private void handleRequest(HttpExchange exchange) throws IOException {
        // Simulación de procesamiento que involucra I/O
        String userId = extractUserId(exchange);
        
        // Cada operación I/O libera el carrier thread
        var user = fetchUserFromDatabase(userId);        // I/O 1
        var preferences = fetchUserPreferences(userId);   // I/O 2  
        var recommendations = callRecommendationAPI(user); // I/O 3
        
        var response = buildResponse(user, preferences, recommendations);
        sendResponse(exchange, response);
    }
}
```

### Migración Práctica de Aplicaciones

```java
// Antes: ThreadPoolTaskExecutor tradicional
@Configuration
@EnableAsync
public class AsyncConfigOld {
    
    @Bean(name = "taskExecutor")
    public ThreadPoolTaskExecutor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(20);
        executor.setMaxPoolSize(200);
        executor.setQueueCapacity(500);
        executor.setThreadNamePrefix("App-Thread-");
        executor.initialize();
        return executor;
    }
}

// Después: Virtual Threads con Spring Boot 3.2+
@Configuration
@EnableAsync
public class AsyncConfigNew {
    
    @Bean(name = "virtualTaskExecutor")
    public AsyncTaskExecutor virtualTaskExecutor() {
        return new TaskExecutorAdapter(Executors.newVirtualThreadPerTaskExecutor());
    }
    
    @Bean
    public TomcatProtocolHandlerCustomizer<?> protocolHandlerVirtualThreadExecutorCustomizer() {
        return protocolHandler -> {
            protocolHandler.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        };
    }
}

// Servicio con operaciones I/O intensivas
@Service
public class OrderProcessingService {
    
    @Async("virtualTaskExecutor")
    public CompletableFuture<OrderResult> processOrderAsync(Order order) {
        // Múltiples operaciones I/O que antes bloquearían threads
        
        // 1. Validar inventario (DB call)
        var inventory = inventoryService.checkAvailability(order.getItems());
        
        // 2. Procesar pago (External API)  
        var payment = paymentService.processPayment(order.getPayment());
        
        // 3. Calcular envío (External API)
        var shipping = shippingService.calculateShipping(order.getAddress());
        
        // 4. Crear factura (DB call)
        var invoice = invoiceService.generateInvoice(order, payment);
        
        // 5. Enviar notificaciones (Email/SMS APIs)
        notificationService.sendOrderConfirmation(order.getCustomer(), invoice);
        
        return CompletableFuture.completedFuture(
            new OrderResult(order.getId(), "PROCESSED", invoice.getId())
        );
    }
    
    // Procesamiento de lotes con virtual threads
    public void processBulkOrders(List<Order> orders) {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            
            // Crear un virtual thread por cada orden
            var futures = orders.stream()
                .map(order -> executor.submit(() -> processOrderSync(order)))
                .collect(Collectors.toList());
            
            // Esperar completación de todos
            futures.forEach(future -> {
                try {
                    future.get();
                } catch (Exception e) {
                    log.error("Error processing order", e);
                }
            });
        }
        
        log.info("Processed {} orders concurrently", orders.size());
    }
}
```

## Pattern Matching Avanzado

### Switch Expressions y Pattern Matching

```java
public class ModernPatternMatching {
    
    // Sealed classes para jerarquías controladas
    public sealed interface PaymentMethod 
        permits CreditCard, DebitCard, PayPal, ApplePay, CryptoCurrency {
    }
    
    public record CreditCard(String number, String cvv, String expiryDate) implements PaymentMethod {}
    public record DebitCard(String number, String pin) implements PaymentMethod {}
    public record PayPal(String email, String password) implements PaymentMethod {}
    public record ApplePay(String deviceId, String touchId) implements PaymentMethod {}
    public record CryptoCurrency(String walletAddress, CryptoType type, BigDecimal amount) implements PaymentMethod {}
    
    // Pattern matching con destructuring
    public BigDecimal calculateProcessingFee(PaymentMethod payment, BigDecimal amount) {
        return switch (payment) {
            case CreditCard(var number, var cvv, var expiry) -> {
                // Lógica específica para tarjeta de crédito
                boolean isPremiumCard = number.startsWith("4111") || number.startsWith("5555");
                yield isPremiumCard ? amount.multiply(new BigDecimal("0.025")) 
                                   : amount.multiply(new BigDecimal("0.030"));
            }
            
            case DebitCard(var number, var pin) -> 
                amount.multiply(new BigDecimal("0.015")); // Menor fee para débito
            
            case PayPal(var email, var password) -> {
                boolean isVerified = payPalService.isVerifiedAccount(email);
                yield isVerified ? amount.multiply(new BigDecimal("0.029"))
                                 : amount.multiply(new BigDecimal("0.040"));
            }
            
            case ApplePay(var deviceId, var touchId) -> 
                amount.multiply(new BigDecimal("0.020")); // Apple Pay premium
            
            case CryptoCurrency(var address, var type, var cryptoAmount) -> 
                switch (type) {
                    case BITCOIN -> amount.multiply(new BigDecimal("0.010"));
                    case ETHEREUM -> amount.multiply(new BigDecimal("0.012"));
                    case STABLECOIN -> amount.multiply(new BigDecimal("0.008"));
                };
        };
    }
    
    // Guard conditions en pattern matching
    public String validatePayment(PaymentMethod payment, BigDecimal amount) {
        return switch (payment) {
            case CreditCard(var number, var cvv, var expiry) 
                when number.length() != 16 -> "Invalid credit card number length";
                
            case CreditCard(var number, var cvv, var expiry) 
                when !isValidExpiryDate(expiry) -> "Credit card expired";
                
            case DebitCard(var number, var pin) 
                when pin.length() != 4 -> "Invalid PIN length";
                
            case PayPal(var email, var password) 
                when !email.contains("@") -> "Invalid email format";
                
            case CryptoCurrency(var address, var type, var cryptoAmount) 
                when cryptoAmount.compareTo(amount) < 0 -> "Insufficient crypto balance";
                
            case PaymentMethod p -> "Valid payment method: " + p.getClass().getSimpleName();
        };
    }
    
    // Pattern matching con instanceof
    public String describeObject(Object obj) {
        return switch (obj) {
            case String s when s.length() > 10 -> "Long string: " + s.substring(0, 10) + "...";
            case String s -> "Short string: " + s;
            case Integer i when i > 100 -> "Large number: " + i;
            case Integer i -> "Small number: " + i;
            case List<?> list when list.isEmpty() -> "Empty list";
            case List<?> list -> "List with " + list.size() + " elements";
            case null -> "Null value";
            default -> "Unknown type: " + obj.getClass().getSimpleName();
        };
    }
}
```

### Aplicaciones Prácticas en Sistemas Empresariales

```java
public class BusinessLogicEngine {
    
    // Jerarquía de eventos de negocio
    public sealed interface BusinessEvent 
        permits OrderEvent, PaymentEvent, InventoryEvent, CustomerEvent {
        
        LocalDateTime getTimestamp();
        String getEventId();
    }
    
    public sealed interface OrderEvent extends BusinessEvent 
        permits OrderCreated, OrderUpdated, OrderCancelled, OrderCompleted {}
    
    public record OrderCreated(String eventId, LocalDateTime timestamp, 
                              Order order, Customer customer) implements OrderEvent {}
    
    public record OrderUpdated(String eventId, LocalDateTime timestamp,
                              Order oldOrder, Order newOrder, String reason) implements OrderEvent {}
    
    public record OrderCompleted(String eventId, LocalDateTime timestamp,
                                Order order, Payment payment, Shipment shipment) implements OrderEvent {}
    
    // Procesador de eventos con pattern matching
    public void processBusinessEvent(BusinessEvent event) {
        var result = switch (event) {
            
            // Eventos de orden con destructuring
            case OrderCreated(var id, var timestamp, var order, var customer) -> {
                log.info("New order created: {} for customer {}", order.getId(), customer.getName());
                
                // Procesos automáticos para nueva orden
                inventoryService.reserveItems(order.getItems());
                emailService.sendOrderConfirmation(customer, order);
                analyticsService.trackOrderCreation(order);
                
                yield ProcessingResult.success("Order created successfully");
            }
            
            case OrderCompleted(var id, var timestamp, var order, var payment, var shipment) -> {
                log.info("Order {} completed with payment {} and shipment {}", 
                        order.getId(), payment.getId(), shipment.getTrackingNumber());
                
                // Procesos de completación
                inventoryService.updateStock(order.getItems());
                loyaltyService.awardPoints(order.getCustomer(), order.getTotal());
                reportingService.updateSalesMetrics(order);
                
                yield ProcessingResult.success("Order completed successfully");
            }
            
            // Eventos de pago
            case PaymentEvent paymentEvent -> switch (paymentEvent) {
                case PaymentProcessed(var id, var timestamp, var payment) 
                    when payment.getAmount().compareTo(new BigDecimal("1000")) > 0 -> {
                    
                    // Pagos grandes requieren verificación adicional
                    fraudDetectionService.analyzeHighValuePayment(payment);
                    yield ProcessingResult.requiresReview("High value payment requires review");
                }
                
                case PaymentFailed(var id, var timestamp, var payment, var reason) -> {
                    orderService.suspendOrder(payment.getOrderId());
                    notificationService.sendPaymentFailureAlert(payment, reason);
                    yield ProcessingResult.failure("Payment processing failed: " + reason);
                }
                
                default -> {
                    standardPaymentProcessor.process(paymentEvent);
                    yield ProcessingResult.success("Payment processed");
                }
            };
            
            // Eventos de inventario
            case InventoryEvent inventoryEvent -> 
                inventoryProcessor.processInventoryEvent(inventoryEvent);
            
            // Eventos de cliente  
            case CustomerEvent customerEvent ->
                customerProcessor.processCustomerEvent(customerEvent);
        };
        
        // Post-procesamiento basado en resultado
        switch (result.getStatus()) {
            case SUCCESS -> metricsService.incrementSuccessCounter(event.getClass().getSimpleName());
            case FAILURE -> alertService.sendProcessingFailureAlert(event, result.getMessage());
            case REQUIRES_REVIEW -> workflowService.createReviewTask(event, result.getMessage());
        }
    }
}
```

## String Templates (Preview Feature)

```java
public class ModernStringHandling {
    
    // String templates para SQL dinámico (¡cuidado con SQL injection!)
    public String buildSecureQuery(String tableName, List<String> columns, 
                                  Map<String, Object> filters) {
        
        // Validación de entrada para seguridad
        var sanitizedTable = validateTableName(tableName);
        var sanitizedColumns = columns.stream()
            .map(this::validateColumnName)
            .collect(Collectors.joining(", "));
        
        var whereClause = filters.entrySet().stream()
            .map(entry -> STR."\{validateColumnName(entry.getKey())} = ?")
            .collect(Collectors.joining(" AND "));
        
        return STR."""
            SELECT \{sanitizedColumns}
            FROM \{sanitizedTable}
            \{whereClause.isEmpty() ? "" : STR."WHERE \{whereClause}"}
            ORDER BY created_date DESC
            """;
    }
    
    // Templates para logging estruturado
    public void logBusinessEvent(BusinessEvent event, ProcessingResult result) {
        var logMessage = switch (event) {
            case OrderCreated(var id, var timestamp, var order, var customer) ->
                STR."""
                Event: ORDER_CREATED
                ID: \{id}
                Timestamp: \{timestamp}
                Order: \{order.getId()}
                Customer: \{customer.getName()} (\{customer.getEmail()})
                Amount: \{order.getTotal()}
                Status: \{result.getStatus()}
                """;
                
            case PaymentProcessed(var id, var timestamp, var payment) ->
                STR."""
                Event: PAYMENT_PROCESSED
                ID: \{id}
                Payment: \{payment.getId()}
                Amount: \{payment.getAmount()}
                Method: \{payment.getMethod().getClass().getSimpleName()}
                Order: \{payment.getOrderId()}
                """;
                
            default -> STR."""
                Event: \{event.getClass().getSimpleName()}
                ID: \{event.getEventId()}
                Timestamp: \{event.getTimestamp()}
                """;
        };
        
        log.info(logMessage);
    }
    
    // Templates para generación de reportes
    public String generateSalesReport(List<SalesSummary> sales, LocalDate reportDate) {
        var totalSales = sales.stream()
            .map(SalesSummary::getAmount)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
            
        var topProduct = sales.stream()
            .max(Comparator.comparing(SalesSummary::getQuantity))
            .map(SalesSummary::getProductName)
            .orElse("N/A");
        
        return STR."""
            ==========================================
            SALES REPORT - \{reportDate.format(DateTimeFormatter.ofPattern("dd/MM/yyyy"))}
            ==========================================
            
            Total Sales: \{NumberFormat.getCurrencyInstance().format(totalSales)}
            Number of Transactions: \{sales.size()}
            Average Transaction: \{NumberFormat.getCurrencyInstance().format(totalSales.divide(new BigDecimal(sales.size()), 2, RoundingMode.HALF_UP))}
            Top Product: \{topProduct}
            
            Transaction Details:
            \{sales.stream()
                .sorted(Comparator.comparing(SalesSummary::getAmount).reversed())
                .limit(10)
                .map(sale -> STR."  • \{sale.getProductName()}: \{NumberFormat.getCurrencyInstance().format(sale.getAmount())} (Qty: \{sale.getQuantity()})")
                .collect(Collectors.joining("\n"))
            }
            
            Generated: \{LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"))}
            ==========================================
            """;
    }
}
```

## Record Patterns y Deconstrucción

```java
public class AdvancedRecordPatterns {
    
    public record Address(String street, String city, String state, String zipCode) {}
    public record Customer(Long id, String name, String email, Address address, CustomerType type) {}
    public record Order(Long id, Customer customer, List<OrderItem> items, LocalDateTime createdAt, OrderStatus status) {}
    public record OrderItem(String productId, String productName, int quantity, BigDecimal price) {}
    
    // Deconstrucción profunda de records anidados
    public BigDecimal calculateShippingCost(Order order) {
        return switch (order) {
            
            // Deconstruir order -> customer -> address
            case Order(var id, Customer(var custId, var name, var email, 
                      Address(var street, var city, var state, var zip), var type), 
                      var items, var createdAt, var status) 
                when "CA".equals(state) || "NY".equals(state) -> {
                
                // Estados con impuestos especiales
                var baseShipping = calculateBaseShipping(items);
                var stateTax = baseShipping.multiply(new BigDecimal("0.08"));
                yield baseShipping.add(stateTax);
            }
            
            // Clientes premium en cualquier estado
            case Order(var id, Customer(var custId, var name, var email, var address, CustomerType.PREMIUM), 
                      var items, var createdAt, var status) -> {
                
                // Envío gratuito para premium
                yield BigDecimal.ZERO;
            }
            
            // Órdenes grandes (más de 5 items)
            case Order(var id, var customer, var items, var createdAt, var status) 
                when items.size() > 5 -> {
                
                var baseShipping = calculateBaseShipping(items);
                var volumeDiscount = baseShipping.multiply(new BigDecimal("0.15"));
                yield baseShipping.subtract(volumeDiscount);
            }
            
            // Caso por defecto
            case Order(var id, var customer, var items, var createdAt, var status) ->
                calculateBaseShipping(items);
        };
    }
    
    // Análisis de datos de ventas con record patterns
    public SalesAnalysis analyzeSales(List<Order> orders) {
        var totalRevenue = BigDecimal.ZERO;
        var customerTypeStats = new EnumMap<CustomerType, Integer>(CustomerType.class);
        var stateDistribution = new HashMap<String, Integer>();
        var topProducts = new HashMap<String, Integer>();
        
        for (var order : orders) {
            switch (order) {
                
                // Análisis por tipo de cliente y ubicación
                case Order(var id, Customer(var custId, var name, var email, 
                          Address(var street, var city, var state, var zip), var type), 
                          var items, var createdAt, OrderStatus.COMPLETED) -> {
                    
                    // Acumular revenue
                    var orderTotal = items.stream()
                        .map(item -> item.price().multiply(new BigDecimal(item.quantity())))
                        .reduce(BigDecimal.ZERO, BigDecimal::add);
                    totalRevenue = totalRevenue.add(orderTotal);
                    
                    // Stats por tipo de cliente
                    customerTypeStats.merge(type, 1, Integer::sum);
                    
                    // Distribución geográfica
                    stateDistribution.merge(state, 1, Integer::sum);
                    
                    // Top productos
                    items.forEach(item -> 
                        topProducts.merge(item.productName(), item.quantity(), Integer::sum)
                    );
                }
                
                // Ignorar órdenes no completadas para análisis de revenue
                default -> { /* skip non-completed orders */ }
            }
        }
        
        return new SalesAnalysis(totalRevenue, customerTypeStats, stateDistribution, topProducts);
    }
    
    // Validación compleja con record patterns
    public List<ValidationError> validateOrder(Order order) {
        var errors = new ArrayList<ValidationError>();
        
        switch (order) {
            
            // Validar dirección incompleta
            case Order(var id, Customer(var custId, var name, var email, 
                      Address(var street, var city, var state, var zip), var type), 
                      var items, var createdAt, var status) 
                when street == null || street.trim().isEmpty() -> {
                
                errors.add(new ValidationError("INVALID_ADDRESS", "Street address is required"));
            }
            
            // Validar email format
            case Order(var id, Customer(var custId, var name, var email, var address, var type), 
                      var items, var createdAt, var status) 
                when email == null || !email.contains("@") -> {
                
                errors.add(new ValidationError("INVALID_EMAIL", "Valid email address is required"));
            }
            
            // Validar items vacíos
            case Order(var id, var customer, var items, var createdAt, var status) 
                when items == null || items.isEmpty() -> {
                
                errors.add(new ValidationError("NO_ITEMS", "Order must contain at least one item"));
            }
            
            // Validar precios negativos en items
            case Order(var id, var customer, var items, var createdAt, var status) 
                when items.stream().anyMatch(item -> item.price().compareTo(BigDecimal.ZERO) <= 0) -> {
                
                errors.add(new ValidationError("INVALID_PRICE", "All items must have positive prices"));
            }
            
            default -> { /* Order is valid */ }
        }
        
        return errors;
    }
}
```

## Performance y Optimización en Java 21

### JIT Compiler Improvements y GC Tuning

```java
public class Java21Performance {
    
    // Aprovechando mejoras del compilador JIT
    @Benchmark
    public void vectorizedOperations() {
        var data = new int[1_000_000];
        
        // El compilador JIT en Java 21 vectoriza automáticamente este loop
        for (int i = 0; i < data.length; i++) {
            data[i] = i * 2 + 1; // Operación vectorizable
        }
        
        // Pattern que se beneficia de auto-vectorización
        var sum = Arrays.stream(data)
            .parallel() // Paralelización + vectorización
            .filter(x -> x % 2 == 0)
            .reduce(0, Integer::sum);
    }
    
    // Optimizaciones para virtual threads
    public class OptimizedWebService {
        
        private final HttpClient httpClient = HttpClient.newBuilder()
            .executor(Executors.newVirtualThreadPerTaskExecutor()) // Virtual threads para HTTP
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        
        // Procesamiento concurrente optimizado
        public CompletableFuture<List<CustomerData>> fetchCustomerData(List<Long> customerIds) {
            
            // Crear batches para optimizar requests
            var batches = Lists.partition(customerIds, 50);
            
            var batchFutures = batches.stream()
                .map(batch -> CompletableFuture.supplyAsync(() -> {
                    // Virtual thread per batch
                    return batch.parallelStream()
                        .map(this::fetchCustomerDataSync)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toList());
                }, Executors.newVirtualThreadPerTaskExecutor()))
                .collect(Collectors.toList());
            
            return CompletableFuture.allOf(batchFutures.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> 
                    batchFutures.stream()
                        .map(CompletableFuture::join)
                        .flatMap(List::stream)
                        .collect(Collectors.toList())
                );
        }
        
        private CustomerData fetchCustomerDataSync(Long customerId) {
            try {
                var request = HttpRequest.newBuilder()
                    .uri(URI.create("https://api.customers.com/v1/customers/" + customerId))
                    .GET()
                    .build();
                
                var response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
                
                if (response.statusCode() == 200) {
                    return objectMapper.readValue(response.body(), CustomerData.class);
                }
                
                return null;
            } catch (Exception e) {
                log.warn("Failed to fetch customer {}: {}", customerId, e.getMessage());
                return null;
            }
        }
    }
    
    // Memory optimization con ZGC
    @Component
    public class MemoryEfficientDataProcessor {
        
        // Configuración JVM para ZGC:
        // -XX:+UnlockExperimentalVMOptions -XX:+UseZGC -XX:+UseLargePages
        // -Xmx8g -XX:SoftMaxHeapSize=7g
        
        public void processLargeDataset(Path dataFile) throws IOException {
            
            // Procesamiento streaming para datasets grandes
            try (var lines = Files.lines(dataFile)) {
                
                var processor = lines
                    .parallel() // Aprovecha múltiples cores
                    .filter(line -> !line.trim().isEmpty())
                    .map(this::parseLine)
                    .filter(Objects::nonNull)
                    .collect(Collectors.groupingBy(
                        DataRecord::getCategory,
                        Collectors.mapping(
                            DataRecord::getValue,
                            Collectors.summarizingDouble(Double::doubleValue)
                        )
                    ));
                
                // ZGC maneja automáticamente la memoria sin pausas largas
                persistResults(processor);
            }
        }
        
        // Pool de objetos para reducir allocations
        private final ObjectPool<StringBuilder> stringBuilderPool = 
            new GenericObjectPool<>(new StringBuilderFactory());
        
        public String formatData(List<DataRecord> records) {
            StringBuilder sb = null;
            try {
                sb = stringBuilderPool.borrowObject();
                sb.setLength(0); // Reset
                
                records.forEach(record -> 
                    sb.append(record.getCategory())
                      .append(": ")
                      .append(record.getValue())
                      .append("\n")
                );
                
                return sb.toString();
                
            } catch (Exception e) {
                log.error("Error formatting data", e);
                return "";
            } finally {
                if (sb != null) {
                    try {
                        stringBuilderPool.returnObject(sb);
                    } catch (Exception e) {
                        log.warn("Error returning StringBuilder to pool", e);
                    }
                }
            }
        }
    }
}
```

## Migración y Adopción

### Guía Práctica de Migración

```java
// 1. Identificar candidatos para virtual threads
public class MigrationStrategy {
    
    // ❌ Antes: Thread pool tradicional
    @Service
    public class OldEmailService {
        
        private final ThreadPoolTaskExecutor executor;
        
        public void sendBulkEmails(List<EmailRequest> requests) {
            requests.forEach(request -> {
                executor.submit(() -> {
                    try {
                        sendEmail(request);
                    } catch (Exception e) {
                        log.error("Failed to send email", e);
                    }
                });
            });
        }
    }
    
    // ✅ Después: Virtual threads
    @Service  
    public class NewEmailService {
        
        public void sendBulkEmails(List<EmailRequest> requests) {
            try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                
                var futures = requests.stream()
                    .map(request -> executor.submit(() -> sendEmail(request)))
                    .collect(Collectors.toList());
                
                // Manejar resultados
                futures.forEach(future -> {
                    try {
                        future.get(30, TimeUnit.SECONDS);
                    } catch (Exception e) {
                        log.error("Failed to send email", e);
                    }
                });
            }
        }
        
        // Structured concurrency (Preview)
        public void sendBulkEmailsStructured(List<EmailRequest> requests) throws Exception {
            
            try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
                
                var subtasks = requests.stream()
                    .map(request -> scope.fork(() -> sendEmail(request)))
                    .collect(Collectors.toList());
                
                scope.join();           // Esperar a que terminen todos
                scope.throwIfFailed();  // Lanzar si alguno falló
                
                // Todos completaron exitosamente
                log.info("All {} emails sent successfully", requests.size());
            }
        }
    }
}

// 2. Refactoring para pattern matching
public class PatternMatchingMigration {
    
    // Antes: instanceof chains
    public String processPaymentOld(Object payment) {
        if (payment instanceof CreditCard) {
            CreditCard cc = (CreditCard) payment;
            return "Processing credit card: " + cc.getLastFourDigits();
        } else if (payment instanceof DebitCard) {
            DebitCard dc = (DebitCard) payment;
            return "Processing debit card: " + dc.getLastFourDigits();
        } else if (payment instanceof PayPal) {
            PayPal pp = (PayPal) payment;
            return "Processing PayPal: " + pp.getEmail();
        } else {
            return "Unknown payment method";
        }
    }
    
    // Después: pattern matching
    public String processPaymentNew(PaymentMethod payment) {
        return switch (payment) {
            case CreditCard(var number, var cvv, var expiry) -> 
                "Processing credit card: ****" + number.substring(number.length() - 4);
            case DebitCard(var number, var pin) -> 
                "Processing debit card: ****" + number.substring(number.length() - 4);
            case PayPal(var email, var password) -> 
                "Processing PayPal: " + email;
            case ApplePay(var deviceId, var touchId) ->
                "Processing Apple Pay: " + deviceId;
            case CryptoCurrency(var address, var type, var amount) ->
                "Processing crypto: " + type + " to " + address.substring(0, 8) + "...";
        };
    }
}
```

## Conclusiones

Java 21 LTS representa un salto evolutivo:

🚀 **Virtual Threads**: Concurrencia masiva sin overhead  
🎯 **Pattern Matching**: Código más expresivo y seguro  
📝 **String Templates**: Interpolación moderna y segura  
🏗️ **Records**: Datos inmutables con menos boilerplate  
⚡ **Performance**: JIT compiler y GC improvements  
🔒 **Sealed Classes**: Jerarquías controladas y exhaustive matching  

Mi experiencia migrando `java-design-patterns-base` y otros proyectos confirma que estas características no son solo syntax sugar - transforman fundamentalmente cómo diseñamos y construimos aplicaciones Java modernas.

---
*¿Estás considerando migrar a Java 21? Comparte tus planes y preocupaciones en los comentarios.*