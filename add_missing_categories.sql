-- ================================================================
-- CATEGORY CLEANUP - ADD MISSING CATEGORIES (Spanish Names)
-- ================================================================
-- Date: 2026-04-16
-- Structure: grupo (parent) → categoria (child)
-- ================================================================

-- STEP 1: CREATE "SIN CLASIFICAR" GRUPO + CATEGORIA
INSERT INTO workspace.superapp.gold_categorias 
(id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion, id_macro, is_active)
VALUES 
(608, 'Sin Clasificar', 'grupo', NULL, 'Productos sin clasificar - requieren revisión manual', CURRENT_TIMESTAMP(), 10, TRUE),
(3026, 'Sin Clasificar - Revision Pendiente', 'categoria', 608, 'Productos que no pudieron ser clasificados automáticamente', CURRENT_TIMESTAMP(), 10, TRUE);

-- STEP 2: ADD COFFEE & TEA CATEGORIES
INSERT INTO workspace.superapp.gold_categorias 
(id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion, id_macro, is_active)
VALUES 
(609, 'Cafe y Te', 'grupo', NULL, 'Café, té e infusiones', CURRENT_TIMESTAMP(), 1, TRUE),
(3027, 'Cafe Molido', 'categoria', 609, 'Café molido en paquete', CURRENT_TIMESTAMP(), 1, TRUE),
(3028, 'Cafe Instantaneo', 'categoria', 609, 'Café instantáneo o soluble', CURRENT_TIMESTAMP(), 1, TRUE),
(3029, 'Cafe en Capsulas', 'categoria', 609, 'Cápsulas de café', CURRENT_TIMESTAMP(), 1, TRUE),
(3030, 'Te e Infusiones', 'categoria', 609, 'Té e infusiones', CURRENT_TIMESTAMP(), 1, TRUE);

-- STEP 3: ADD SPIRITS SUBCATEGORIES
INSERT INTO workspace.superapp.gold_categorias 
(id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion, id_macro, is_active)
VALUES 
(3031, 'Ginebra', 'categoria', 401, 'Gin y ginebra', CURRENT_TIMESTAMP(), 2, TRUE),
(3032, 'Vodka', 'categoria', 401, 'Vodka', CURRENT_TIMESTAMP(), 2, TRUE),
(3033, 'Vermut', 'categoria', 401, 'Vermouth y vermut', CURRENT_TIMESTAMP(), 2, TRUE);

-- STEP 4: NON-FOOD ESSENTIALS
INSERT INTO workspace.superapp.gold_categorias 
(id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion, id_macro, is_active)
VALUES 
(610, 'Cuidado Personal Adicional', 'grupo', NULL, 'Categorías adicionales de cuidado personal', CURRENT_TIMESTAMP(), 6, TRUE),
(611, 'Bazar', 'grupo', NULL, 'Bazar y artículos para el hogar', CURRENT_TIMESTAMP(), 9, TRUE),
(3034, 'Cuidado del Cabello', 'categoria', 610, 'Productos para el cuidado del cabello', CURRENT_TIMESTAMP(), 6, TRUE),
(3035, 'Bazar y Cocina', 'categoria', 611, 'Utensilios de cocina y bazar', CURRENT_TIMESTAMP(), 9, TRUE),
(3036, 'Electronica y Cables', 'categoria', 611, 'Cables, cargadores, auriculares', CURRENT_TIMESTAMP(), 9, TRUE),
(3037, 'Iluminacion', 'categoria', 611, 'Lámparas, bombillas, LED', CURRENT_TIMESTAMP(), 9, TRUE),
(3038, 'Textiles para el Hogar', 'categoria', 611, 'Textiles del hogar', CURRENT_TIMESTAMP(), 9, TRUE);

-- STEP 5: SPECIALIZED CATEGORIES
INSERT INTO workspace.superapp.gold_categorias 
(id_categoria, nombre, nivel, parent_id, descripcion, fecha_creacion, id_macro, is_active)
VALUES 
(3039, 'Jardineria', 'categoria', 611, 'Productos de jardinería', CURRENT_TIMESTAMP(), 9, TRUE),
(3040, 'Libreria y Papeleria', 'categoria', 611, 'Librería y papelería', CURRENT_TIMESTAMP(), 9, TRUE),
(3041, 'Articulos para Bebe', 'categoria', 610, 'Artículos para bebés', CURRENT_TIMESTAMP(), 6, TRUE),
(3042, 'Jugueteria', 'categoria', 611, 'Juguetes', CURRENT_TIMESTAMP(), 9, TRUE),
(3043, 'Cosmetica y Belleza', 'categoria', 610, 'Cosméticos y belleza', CURRENT_TIMESTAMP(), 6, TRUE),
(3044, 'Herramientas y Ferreteria', 'categoria', 611, 'Herramientas y ferretería', CURRENT_TIMESTAMP(), 9, TRUE),
(3045, 'Automotor', 'categoria', 611, 'Productos automotores', CURRENT_TIMESTAMP(), 9, TRUE),
(3046, 'Vitaminas y Suplementos', 'categoria', 610, 'Vitaminas y suplementos', CURRENT_TIMESTAMP(), 6, TRUE);

-- VERIFICATION
SELECT nivel, id_categoria, nombre, parent_id 
FROM workspace.superapp.gold_categorias 
WHERE id_categoria >= 608 ORDER BY nivel DESC, id_categoria;
