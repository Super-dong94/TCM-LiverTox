import { expect, test } from '@playwright/test';

const baseURL = process.env.TOXHERB_E2E_URL || 'http://127.0.0.1:7860';

test('home database visuals and responsive layout', async ({ page }) => {
  await page.goto(baseURL);
  await expect(page.locator('#homeView')).toBeVisible();
  await expect(page.locator('#homeTitle')).toContainText('中药肝毒性多层级');
  await expect(page.locator('#stat-compounds')).not.toHaveText('--', { timeout: 30000 });
  await expect(page.locator('#databaseVisualPanel')).toBeVisible({ timeout: 30000 });
  await page.click('#langEnBtn');
  await expect(page.locator('#homeTitle')).toContainText('ToxHerb: A Multilevel');
  await expect(page.locator('#currentTimeLabel')).toHaveText('Current Time');
  await expect(page.locator('#themeToggleBtn')).toContainText('Dark Mode');
  await expect(page.locator('#homeSearchTitle')).toHaveText('Search');
  await expect(page.locator('#databaseInfoTitle')).toHaveText('Database Overview');
  await expect(page.locator('#homeInputType option[value="formula"]')).toHaveText('Formula');
  await expect(page.locator('#homeInputType option[value="herb"]')).toHaveText('Herb');
  await expect(page.locator('#homeInputType option[value="compound"]')).toHaveText('Ingredient');
  await page.setViewportSize({ width: 390, height: 844 });
  const bodyBox = await page.locator('body').boundingBox();
  expect(bodyBox?.width).toBeLessThanOrEqual(390);
});

test('knowledge search renders charts without javascript errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(baseURL);
  await page.click('#langEnBtn');
  await page.selectOption('#homeInputType', 'target');
  await page.fill('#homeInputContent', 'CYP3A4');
  await page.click('#homeSearchBtn');
  await expect(page.locator('#visualizationArea')).toBeVisible({ timeout: 60000 });
  await expect(page.locator('#chartGrid .chart-card').first()).toBeVisible();
  const visualActions = page.locator('#chartGrid .chart-card .chart-actions').first();
  await expect(visualActions).toContainText('PNG');
  await expect(visualActions).toContainText('PDF');
  await expect(visualActions).not.toContainText('SVG');
  await expect(visualActions).not.toContainText('JSON');
  await expect(page.locator('#reportGroupTabs')).toContainText('Match Traceability');
  await expect(page.locator('#tableContent')).toBeVisible();
  await page.click('#langZhBtn');
  await expect(page.locator('#inputContent')).toHaveValue('CYP3A4');
  await expect(page.locator('#reportGroupTabs')).toContainText('匹配溯源');
  expect(errors).toEqual([]);
});

test('toxicity overview charts use dedicated desktop and mobile grids', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(baseURL);
  await page.click('#langEnBtn');
  await page.evaluate(() => {
    (window as any).showPlatformView();
    const compounds = [
      {
        CID: '1',
        ChemicalName: 'Emodin',
        Source_Herbs: 'He Shou Wu',
        Class: 'Anthraquinones',
        Superclass: 'Phenylpropanoids and polyketides',
        Pathway: 'Shikimates and phenylpropanoids',
        IntoBlood: 1,
        Is_Absorbed: 1,
        Max_Tox_Prob: 0.91,
      },
      {
        CID: '2',
        ChemicalName: 'Physcion',
        Source_Herbs: 'He Shou Wu',
        Class: 'Anthraquinones',
        Superclass: 'Phenylpropanoids and polyketides',
        Pathway: 'Polyketides',
        IntoBlood: 1,
        Is_Absorbed: 1,
        Max_Tox_Prob: 0.84,
      },
    ];
    (window as any).generateReport('He Shou Wu', 'herb', true, {
      summary: {
        max_toxic_probability: 0.91,
        raw_max_toxic_probability: 0.91,
        risk_level: 'high',
        risk_label: 'High Risk',
        calibration_status: 'missing',
        high_toxic_count: 2,
        toxic_threshold: 0.5,
      },
      probabilities: { cell: 0.86, animal: 0.79, clinical: 0.72 },
      compounds,
      high_risk_compounds: compounds,
      trace_details: {
        formula_herbs: [
          { 'Formula.Chinese.name': 'Sample Formula A', 'Herb.Chinese.name': 'He Shou Wu' },
          { 'Formula.Chinese.name': 'Sample Formula B', 'Herb.Chinese.name': 'He Shou Wu' },
        ],
      },
      toxicity_details: {
        toxic_compounds: compounds,
        toxic_targets: [
          { ChemicalName: 'Emodin', Symbol: 'CYP3A4', ENTREZID: '1576', Max_Tox_Prob: 0.91 },
          { ChemicalName: 'Physcion', Symbol: 'ABCB1', ENTREZID: '5243', Max_Tox_Prob: 0.84 },
        ],
      },
      class_counts: [{ name: 'Anthraquinones', count: 2 }],
      superclass_counts: [{ name: 'Phenylpropanoids and polyketides', count: 2 }],
      pathway_family_counts: [{ name: 'Polyketides', count: 2 }],
      target_counts: [{ name: 'CYP3A4', count: 2 }],
      pathway_counts: [{ name: 'Drug metabolism', count: 2 }],
      go_counts: [{ name: 'oxidative stress response', count: 2 }],
    });
  });
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Toxicity Analysis Overview' }).click();

  const columnCount = (selector: string) => page.locator(selector).evaluate((element) => {
    const columns = getComputedStyle(element).gridTemplateColumns;
    return columns === 'none' ? 0 : columns.split(/\s+/).filter(Boolean).length;
  });
  await expect(page.locator('.toxicity-overview-network-grid')).toBeVisible();
  await expect(page.locator('.toxicity-overview-stat-grid')).toBeVisible();
  await expect(page.locator('.toxicity-overview-network-grid .chart-card.network-chart')).toHaveCount(4);
  await expect(page.locator('.toxicity-overview-stat-grid .chart-card')).toHaveCount(6);
  await expect(page.locator('.toxicity-overview-network-grid .chart-card-title')).toHaveText([
    'Herb-Formula Network',
    'Herb-Ingredient Network',
    'Herb-Systemically Absorbed Ingredient Network',
    'Herb-Toxic Ingredient Network',
  ]);
  await expect(page.locator('.toxicity-overview-network-grid')).not.toContainText('Query Entity');
  const graphLayoutOptions = await page.locator('.toxicity-overview-network-grid .chart-canvas').evaluateAll((canvases) =>
    canvases.map((canvas) => {
      const chart = (window as any).echarts.getInstanceByDom(canvas);
      const series = chart.getOption().series[0];
      return {
        layout: series.layout,
        nodesHaveCoordinates: series.data.every((node: Record<string, unknown>) =>
          typeof node.x === 'number' && typeof node.y === 'number'
        ),
      };
    })
  );
  expect(graphLayoutOptions).toHaveLength(4);
  expect(graphLayoutOptions.every(option => option.layout === 'none')).toBe(true);
  expect(graphLayoutOptions.every(option => option.nodesHaveCoordinates)).toBe(true);
  const reportActions = page.locator('.toxicity-overview-network-grid .chart-actions').first();
  await expect(reportActions).toContainText('PNG');
  await expect(reportActions).toContainText('PDF');
  await expect(reportActions).not.toContainText('SVG');
  await expect(reportActions).not.toContainText('JSON');
  const pngDownload = page.waitForEvent('download');
  await reportActions.getByText('PNG').click();
  expect((await pngDownload).suggestedFilename()).toMatch(/\.png$/);
  const pdfDownload = page.waitForEvent('download');
  await reportActions.getByText('PDF').click();
  expect((await pdfDownload).suggestedFilename()).toMatch(/\.pdf$/);
  expect(await columnCount('.toxicity-overview-network-grid')).toBe(4);
  expect(await columnCount('.toxicity-overview-stat-grid')).toBe(3);
  expect(await page.locator('.toxicity-overview-network-grid .chart-card.network-chart').evaluateAll(cards =>
    cards.map(card => getComputedStyle(card).gridColumnEnd)
  )).toEqual(['auto', 'auto', 'auto', 'auto']);
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Targets Associated with Toxic Ingredients' }).click();
  await expect(page.locator('#tableContent .chart-card-title')).toContainText('Herb-Toxic Ingredient-Target Network');
  await expect(page.locator('#tableContent')).not.toContainText('Query Entity');
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Toxicity Analysis Overview' }).click();

  await page.setViewportSize({ width: 390, height: 844 });
  expect(await columnCount('.toxicity-overview-network-grid')).toBe(1);
  expect(await columnCount('.toxicity-overview-stat-grid')).toBe(1);
  const bodyBox = await page.locator('body').boundingBox();
  expect(bodyBox?.width).toBeLessThanOrEqual(390);
});

test('toxicity network titles reflect prediction entry type', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(baseURL);
  await page.click('#langEnBtn');

  async function renderSyntheticReport(inputContent: string, typeValue: string) {
    await page.evaluate(({ inputContent, typeValue }) => {
      (window as any).showPlatformView();
      const compounds = [
        {
          CID: '1',
          ChemicalName: 'Emodin',
          Source_Herbs: 'He Shou Wu',
          Class: 'Anthraquinones',
          Superclass: 'Phenylpropanoids and polyketides',
          Pathway: 'Polyketides',
          IntoBlood: 1,
          Is_Absorbed: 1,
          Max_Tox_Prob: 0.91,
        },
        {
          CID: '2',
          ChemicalName: 'Physcion',
          Source_Herbs: 'He Shou Wu',
          Class: 'Anthraquinones',
          Superclass: 'Phenylpropanoids and polyketides',
          Pathway: 'Polyketides',
          IntoBlood: 1,
          Is_Absorbed: 1,
          Max_Tox_Prob: 0.84,
        },
      ];
      (window as any).generateReport(inputContent, typeValue, true, {
        summary: {
          max_toxic_probability: 0.91,
          raw_max_toxic_probability: 0.91,
          risk_level: 'high',
          risk_label: 'High Risk',
          calibration_status: 'missing',
          high_toxic_count: 2,
          toxic_threshold: 0.5,
        },
        probabilities: { cell: 0.86, animal: 0.79, clinical: 0.72 },
        compounds,
        high_risk_compounds: compounds,
        trace_details: {
          formula_herbs: [
            { 'Formula.Chinese.name': 'Sample Formula A', 'Herb.Chinese.name': 'He Shou Wu' },
            { 'Formula.Chinese.name': 'Sample Formula B', 'Herb.Chinese.name': 'He Shou Wu' },
          ],
        },
        toxicity_details: {
          toxic_compounds: compounds,
          toxic_targets: [
            { ChemicalName: 'Emodin', Symbol: 'CYP3A4', ENTREZID: '1576', Max_Tox_Prob: 0.91 },
          ],
        },
        class_counts: [{ name: 'Anthraquinones', count: 2 }],
        superclass_counts: [{ name: 'Phenylpropanoids and polyketides', count: 2 }],
        pathway_family_counts: [{ name: 'Polyketides', count: 2 }],
        target_counts: [{ name: 'CYP3A4', count: 2 }],
        pathway_counts: [{ name: 'Drug metabolism', count: 2 }],
        go_counts: [{ name: 'oxidative stress response', count: 2 }],
      });
    }, { inputContent, typeValue });
    await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Toxicity Analysis Overview' }).click();
  }

  await renderSyntheticReport('Sample Formula A', 'formula');
  await expect(page.locator('.toxicity-overview-network-grid .chart-card-title')).toHaveText([
    'Herb-Formula Network',
    'Herb-Ingredient Network',
    'Herb-Systemically Absorbed Ingredient Network',
    'Herb-Toxic Ingredient Network',
  ]);
  await expect(page.locator('.toxicity-overview-network-grid')).not.toContainText('Query Entity');
  const formulaOverviewText = await page.locator('.toxicity-overview-network-grid .chart-card-title').allInnerTexts();
  expect(formulaOverviewText.join(' ')).not.toContain('(');
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Targets Associated with Toxic Ingredients' }).click();
  await expect(page.locator('#tableContent .chart-card-title')).toContainText('Herb-Toxic Ingredient-Target Network');

  await renderSyntheticReport('Emodin', 'compound');
  await expect(page.locator('.toxicity-overview-network-grid .chart-card-title')).toHaveText([
    'Ingredient-Formula Network',
    'Ingredient-Associated Ingredient Network',
    'Ingredient-Systemically Absorbed Ingredient Network',
    'Ingredient-Toxic Ingredient Network',
  ]);
  await expect(page.locator('.toxicity-overview-network-grid')).not.toContainText('Query Entity');
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: 'Targets Associated with Toxic Ingredients' }).click();
  await expect(page.locator('#tableContent .chart-card-title')).toContainText('Ingredient-Toxic Ingredient-Target Network');
  await expect(page.locator('#tableContent')).not.toContainText('Query Entity');
});

test('multi-herb toxicity networks keep separate herb root nodes', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(baseURL);
  await page.evaluate(() => {
    (window as any).showPlatformView();
    const compounds = [
      {
        CID: '101',
        ChemicalName: 'safflor yellow A',
        'Herb.Chinese.name': '红花',
        Chinese_medicine_group: '红花_黄芪',
        Class: 'Flavonoids',
        Superclass: 'Phenylpropanoids and polyketides',
        Pathway: 'Flavonoids',
        IntoBlood: 1,
        Is_Absorbed: 1,
        Max_Tox_Prob: 0.91,
      },
      {
        CID: '102',
        ChemicalName: 'astragaloside IV',
        'Herb.Chinese.name': '黄芪',
        Chinese_medicine_group: '红花_黄芪',
        Class: 'Triterpenoids',
        Superclass: 'Lipids and lipid-like molecules',
        Pathway: 'Terpenoids',
        IntoBlood: 1,
        Is_Absorbed: 1,
        Max_Tox_Prob: 0.88,
      },
      {
        CID: '103',
        ChemicalName: 'shared marker',
        Source_Herbs: '红花、黄芪',
        Class: 'Phenolics',
        Superclass: 'Organic oxygen compounds',
        Pathway: 'Phenylpropanoids',
        IntoBlood: 1,
        Is_Absorbed: 1,
        Max_Tox_Prob: 0.86,
      },
    ];
    (window as any).generateReport('红花_黄芪', 'herb', true, {
      query: ['红花', '黄芪'],
      summary: {
        max_toxic_probability: 0.91,
        raw_max_toxic_probability: 0.91,
        risk_level: 'high',
        risk_label: '高风险',
        calibration_status: 'missing',
        high_toxic_count: 3,
        toxic_threshold: 0.85,
      },
      probabilities: { cell: 0.86, animal: 0.91, clinical: 0.72 },
      compounds,
      high_risk_compounds: compounds,
      trace_details: {
        formula_herbs: [
          { 'Formula.Chinese.name': '含红花方剂', 'Herb.Chinese.name': '红花' },
          { 'Formula.Chinese.name': '含黄芪方剂', 'Herb.Chinese.name': '黄芪' },
        ],
        herb_compounds: [
          { 'Herb.Chinese.name': '红花', ChemicalName: 'safflor yellow A' },
          { 'Herb.Chinese.name': '黄芪', ChemicalName: 'astragaloside IV' },
        ],
      },
      toxicity_details: {
        toxic_compounds: compounds,
        toxic_targets: [
          { ChemicalName: 'safflor yellow A', 'Herb.Chinese.name': '红花', Chinese_medicine_group: '红花_黄芪', Symbol: 'CYP3A4', ENTREZID: '1576', Max_Tox_Prob: 0.91 },
          { ChemicalName: 'astragaloside IV', 'Herb.Chinese.name': '黄芪', Chinese_medicine_group: '红花_黄芪', Symbol: 'ABCB1', ENTREZID: '5243', Max_Tox_Prob: 0.88 },
          { ChemicalName: 'shared marker', Source_Herbs: '红花、黄芪', Symbol: 'NFE2L2', ENTREZID: '4780', Max_Tox_Prob: 0.86 },
        ],
      },
      class_counts: [{ name: 'Flavonoids', count: 1 }, { name: 'Triterpenoids', count: 1 }],
      superclass_counts: [{ name: 'Phenylpropanoids and polyketides', count: 1 }],
      pathway_family_counts: [{ name: 'Flavonoids', count: 1 }],
      target_counts: [{ name: 'CYP3A4', count: 1 }, { name: 'ABCB1', count: 1 }],
      pathway_counts: [{ name: 'Drug metabolism', count: 1 }],
      go_counts: [{ name: 'oxidative stress response', count: 1 }],
    });
  });

  const readFirstTraceGraph = async () => page.locator('#chartGrid .chart-canvas').first().evaluate((canvas) => {
    const chart = (window as any).echarts.getInstanceByDom(canvas);
    const series = chart.getOption().series[0];
    return series.data.map((node: Record<string, unknown>) => String(node.name || ''));
  });

  await page.locator('#reportGroupTabs .report-group-tab', { hasText: '匹配溯源' }).click();
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: '方剂' }).click();
  const traceFormulaNames = await readFirstTraceGraph();
  expect(traceFormulaNames).toContain('红花');
  expect(traceFormulaNames).toContain('黄芪');
  expect(traceFormulaNames).not.toContain('红花_黄芪');
  expect(traceFormulaNames).not.toContain('红花?黄芪');
  expect(traceFormulaNames).not.toContain('红花 / 黄芪');

  await page.locator('#reportSubTabs .report-sub-tab', { hasText: '中药' }).click();
  const traceHerbNames = await readFirstTraceGraph();
  expect(traceHerbNames).toContain('红花');
  expect(traceHerbNames).toContain('黄芪');
  expect(traceHerbNames).not.toContain('红花_黄芪');
  expect(traceHerbNames).not.toContain('红花?黄芪');
  expect(traceHerbNames).not.toContain('红花 / 黄芪');

  await page.locator('#reportGroupTabs .report-group-tab', { hasText: '肝毒性分析' }).click();
  await page.locator('#reportSubTabs .report-sub-tab', { hasText: '毒性分析概览' }).click();

  const overviewGraphs = await page.locator('.toxicity-overview-network-grid .chart-canvas').evaluateAll((canvases) =>
    canvases.map((canvas) => {
      const chart = (window as any).echarts.getInstanceByDom(canvas);
      const series = chart.getOption().series[0];
      const nodes = series.data.map((node: Record<string, unknown>) => ({
        id: String(node.id || ''),
        name: String(node.name || ''),
        category: String(node.category || ''),
      }));
      const nodeNameById = new Map(nodes.map((node: { id: string; name: string }) => [node.id, node.name]));
      const links = series.links.map((link: Record<string, unknown>) => ({
        source: nodeNameById.get(String(link.source || '')) || String(link.source || ''),
        target: nodeNameById.get(String(link.target || '')) || String(link.target || ''),
        relation: String(link.relation || ''),
      }));
      return { nodes, links };
    })
  );
  expect(overviewGraphs).toHaveLength(4);
  const allOverviewNames = overviewGraphs.flatMap(graph => graph.nodes.map(node => node.name));
  expect(allOverviewNames).toContain('红花');
  expect(allOverviewNames).toContain('黄芪');
  expect(allOverviewNames).not.toContain('红花 / 黄芪');
  expect(allOverviewNames).not.toContain('红花、黄芪');
  expect(allOverviewNames).not.toContain('红花_黄芪');
  expect(allOverviewNames).not.toContain('红花?黄芪');
  expect(allOverviewNames).not.toContain('Query Entity');

  const ingredientGraph = overviewGraphs[1];
  expect(ingredientGraph.links).toEqual(expect.arrayContaining([
    expect.objectContaining({ source: '红花', target: 'safflor yellow A' }),
    expect.objectContaining({ source: '黄芪', target: 'astragaloside IV' }),
    expect.objectContaining({ source: '红花', target: 'shared marker' }),
    expect.objectContaining({ source: '黄芪', target: 'shared marker' }),
  ]));

  await page.locator('#reportSubTabs .report-sub-tab', { hasText: '毒性成分对应的靶标' }).click();
  const targetGraph = await page.locator('#tableContent .chart-canvas').first().evaluate((canvas) => {
    const chart = (window as any).echarts.getInstanceByDom(canvas);
    const series = chart.getOption().series[0];
    const nodes = series.data.map((node: Record<string, unknown>) => ({
      id: String(node.id || ''),
      name: String(node.name || ''),
      category: String(node.category || ''),
    }));
    const nodeNameById = new Map(nodes.map((node: { id: string; name: string }) => [node.id, node.name]));
    const links = series.links.map((link: Record<string, unknown>) => ({
      source: nodeNameById.get(String(link.source || '')) || String(link.source || ''),
      target: nodeNameById.get(String(link.target || '')) || String(link.target || ''),
      relation: String(link.relation || ''),
    }));
    return { nodes, links };
  });
  const targetGraphNames = targetGraph.nodes.map(node => node.name);
  expect(targetGraphNames).toContain('红花');
  expect(targetGraphNames).toContain('黄芪');
  expect(targetGraphNames).not.toContain('红花 / 黄芪');
  expect(targetGraphNames).not.toContain('红花_黄芪');
  expect(targetGraphNames).not.toContain('红花?黄芪');
  expect(targetGraph.links).toEqual(expect.arrayContaining([
    expect.objectContaining({ source: '红花', target: 'safflor yellow A' }),
    expect.objectContaining({ source: '黄芪', target: 'astragaloside IV' }),
    expect.objectContaining({ source: '红花', target: 'shared marker' }),
    expect.objectContaining({ source: '黄芪', target: 'shared marker' }),
  ]));
});
